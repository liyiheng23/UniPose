# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os.path as osp
import sys
import warnings
from argparse import ArgumentParser
from collections import abc
from typing import Any, Optional, Union, Type, List
import importlib
from importlib import import_module
from importlib.util import find_spec
from packaging.version import parse
from collections import defaultdict



def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

class LazyObject:
    """LazyObject is used to lazily initialize the imported module during
    parsing the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models
        >>> import mmcls.datasets
        >>> import mmcls

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = lazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')
        >>> # import mmcls.models; import mmcls.datasets; import mmcls
        >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

    ``LazyObject`` records all module information and will be further
    referenced by the configuration file.

    Args:
        module (str or list or tuple): The module name to be imported.
        imported (str, optional): The imported module name. Defaults to None.
        location (str, optional): The filename and line number of the imported
            module statement happened.
    """

    def __init__(self,
                 module: Union[str, list, tuple],
                 imported: Optional[str] = None,
                 location: Optional[str] = None):
        if not isinstance(module, str) and not is_seq_of(module, str):
            raise TypeError('module should be `str`, `list`, or `tuple`'
                            f'but got {type(module)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._module: Union[str, list, tuple] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError('imported should be `str` or None, but got '
                            f'{type(imported)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._imported = imported
        self.location = location

    def build(self) -> Any:
        """Return imported object.

        Returns:
            Any: Imported object
        """
        if isinstance(self._module, str):
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f'Failed to import {self._module} '
                              f'in {self.location} for {e}')

            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(
                        f'Failed to import {self._imported} '
                        f'from {self._module} in {self.location}')

            return module
        else:
            # import xxx.xxx
            # import xxx.yyy
            # import xxx.zzz
            # return imported xxx
            try:
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split('.')[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f'Failed to import {self.module} '
                              f'in {self.location} for {e}')

    @property
    def module(self):
        if isinstance(self._module, str):
            return self._module
        return self._module[0].split('.')[0]

    def __call__(self, *args, **kwargs):
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyObject(self._module, self._imported, self.location)

    def __getattr__(self, name):
        # Cannot locate the line number of the getting attribute.
        # Therefore only record the filename.
        if self.location is not None:
            location = self.location.split(', line')[0]
        else:
            location = self.location
        return LazyAttr(name, self, location)

    def __str__(self) -> str:
        if self._imported is not None:
            return self._imported
        return self.module

    __repr__ = __str__

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyObject will return a `__getstate__` LazyObject` or `__setstate__`
    # LazyObject.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

class LazyAttr:
    """The attribute of the LazyObject.

    When parsing the configuration file, the imported syntax will be
    parsed as the assignment ``LazyObject``. During the subsequent parsing
    process, users may reference the attributes of the LazyObject.
    To ensure that these attributes also contain information needed to
    reconstruct the attribute itself, LazyAttr was introduced.

    Examples:
        >>> models = LazyObject(['mmdet.models'])
        >>> model = dict(type=models.RetinaNet)
        >>> print(type(model['type']))  # <class 'mmengine.config.lazy.LazyAttr'>
        >>> print(model['type'].build())  # <class 'mmdet.models.detectors.retinanet.RetinaNet'>
    """  # noqa: E501

    def __init__(self,
                 name: str,
                 source: Union['LazyObject', 'LazyAttr'],
                 location=None):
        self.name = name
        self.source: Union[LazyAttr, LazyObject] = source

        if isinstance(self.source, LazyObject):
            if isinstance(self.source._module, str):
                if self.source._imported is None:
                    # source code:
                    # from xxx.yyy import zzz
                    # equivalent code:
                    # zzz = LazyObject('xxx.yyy', 'zzz')
                    # The source code of get attribute:
                    # eee = zzz.eee
                    # Then, `eee._module` should be "xxx.yyy.zzz"
                    self._module = self.source._module
                else:
                    # source code:
                    # import xxx.yyy as zzz
                    # equivalent code:
                    # zzz = LazyObject('xxx.yyy')
                    # The source code of get attribute:
                    # eee = zzz.eee
                    # Then, `eee._module` should be "xxx.yyy"
                    self._module = f'{self.source._module}.{self.source}'
            else:
                # The source code of LazyObject should be
                # 1. import xxx.yyy
                # 2. import xxx.zzz
                # Equivalent to
                # xxx = LazyObject(['xxx.yyy', 'xxx.zzz'])

                # The source code of LazyAttr should be
                # eee = xxx.eee
                # Then, eee._module = xxx
                self._module = str(self.source)
        elif isinstance(self.source, LazyAttr):
            # 1. import xxx
            # 2. zzz = xxx.yyy.zzz

            # Equivalent to:
            # xxx = LazyObject('xxx')
            # zzz = xxx.yyy.zzz
            # zzz._module = xxx.yyy._module + zzz.name
            self._module = f'{self.source._module}.{self.source.name}'
        self.location = location

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs: Any) -> Any:
        raise RuntimeError()

    def __getattr__(self, name: str) -> 'LazyAttr':
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self) -> Any:
        """Return the attribute of the imported object.

        Returns:
            Any: attribute of the imported object.
        """
        obj = self.source.build()
        try:
            return getattr(obj, self.name)
        except AttributeError:
            raise ImportError(f'Failed to import {self.module}.{self.name} in '
                              f'{self.location}')
        except ImportError as e:
            raise e

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyAttr will return a `__getstate__` LazyAttr` or `__setstate__`
    # LazyAttr.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

def _lazy2string(cfg_dict, dict_type=None):
    if isinstance(cfg_dict, dict):
        dict_type = dict_type or type(cfg_dict)
        return dict_type(
            {k: _lazy2string(v, dict_type)
             for k, v in dict.items(cfg_dict)})
    elif isinstance(cfg_dict, (tuple, list)):
        return type(cfg_dict)(_lazy2string(v, dict_type) for v in cfg_dict)
    elif isinstance(cfg_dict, (LazyAttr, LazyObject)):
        return f'{cfg_dict.module}.{str(cfg_dict)}'
    else:
        return cfg_dict

def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)

def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f'Failed to import {imp}')
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported

class ConfigParsingError(RuntimeError):
    """Raise error when failed to parse pure Python style config files."""

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == self.key):
            return None
        else:
            return node

def add_args(parser: ArgumentParser,
             cfg: dict,
             prefix: str = '') -> ArgumentParser:
    """Add config fields into argument parser.

    Args:
        parser (ArgumentParser): Argument parser.
        cfg (dict): Config dictionary.
        prefix (str, optional): Prefix of parser argument.
            Defaults to ''.

    Returns:
        ArgumentParser: Argument parser containing config fields.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument(
                '--' + prefix + k, type=type(next(iter(v))), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
            raise NotImplementedError
    return parser

def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    import importlib.util

    from pkg_resources import DistributionNotFound, get_distribution

    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    try:
        pkg = get_distribution(package)
    except DistributionNotFound as e:
        # if the package is not installed, package path set in PYTHONPATH
        # can be detected by `find_spec`
        spec = importlib.util.find_spec(package)
        if spec is not None:
            if spec.origin is not None:
                return osp.dirname(spec.origin)
            else:
                # `get_installed_path` cannot get the installed path of
                # namespace packages
                raise RuntimeError(
                    f'{package} is a namespace package, which is invalid '
                    'for `get_install_path`')
        else:
            raise e

    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))

def package2module(package: str):
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.
    """
    from pkg_resources import get_distribution
    pkg = get_distribution(package)
    if pkg.has_metadata('top_level.txt'):
        module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
        return module_name
    else:
        raise ValueError(f'can not infer the module name of {package}')

class ImportTransformer(ast.NodeTransformer):
    """Convert the import syntax to the assignment of
    :class:`mmengine.config.LazyObject` and preload the base variable before
    parsing the configuration file.

    Since you are already looking at this part of the code, I believe you must
    be interested in the mechanism of the ``lazy_import`` feature of
    :class:`Config`. In this docstring, we will dive deeper into its
    principles.

    Most of OpenMMLab users maybe bothered with that:

        * In most of popular IDEs, they cannot navigate to the source code in
          configuration file
        * In most of popular IDEs, they cannot jump to the base file in current
          configuration file, which is much painful when the inheritance
          relationship is complex.

    In order to solve this problem, we introduce the ``lazy_import`` mode.

    A very intuitive idea for solving this problem is to import the module
    corresponding to the "type" field using the ``import`` syntax. Similarly,
    we can also ``import`` base file.

    However, this approach has a significant drawback. It requires triggering
    the import logic to parse the configuration file, which can be
    time-consuming. Additionally, it implies downloading numerous dependencies
    solely for the purpose of parsing the configuration file.
    However, it's possible that only a portion of the config will actually be
    used. For instance, the package used in the ``train_pipeline`` may not
    be necessary for an evaluation task. Forcing users to download these
    unused packages is not a desirable solution.

    To avoid this problem, we introduce :class:`mmengine.config.LazyObject` and
    :class:`mmengine.config.LazyAttr`. Before we proceed with further
    explanations, you may refer to the documentation of these two modules to
    gain an understanding of their functionalities.

    Actually, one of the functions of ``ImportTransformer`` is to hack the
    ``import`` syntax. It will replace the import syntax
    (exclude import the base files) with the assignment of ``LazyObject``.

    As for the import syntax of the base file, we cannot lazy import it since
    we're eager to merge the fields of current file and base files. Therefore,
    another function of the ``ImportTransformer`` is to collaborate with
    ``Config._parse_lazy_import`` to parse the base files.

    Args:
        global_dict (dict): The global dict of the current configuration file.
            If we divide ordinary Python syntax into two parts, namely the
            import section and the non-import section (assuming a simple case
            with imports at the beginning and the rest of the code following),
            the variables generated by the import statements are stored in
            global variables for subsequent code use. In this context,
            the ``global_dict`` represents the global variables required when
            executing the non-import code. ``global_dict`` will be filled
            during visiting the parsed code.
        base_dict (dict): All variables defined in base files.

            Examples:
                >>> from mmengine.config import read_base
                >>>
                >>>
                >>> with read_base():
                >>>     from .._base_.default_runtime import *
                >>>     from .._base_.datasets.coco_detection import dataset

            In this case, the base_dict will be:

            Examples:
                >>> base_dict = {
                >>>     '.._base_.default_runtime': ...
                >>>     '.._base_.datasets.coco_detection': dataset}

            and `global_dict` will be updated like this:

            Examples:
                >>> global_dict.update(base_dict['.._base_.default_runtime'])  # `import *` means update all data
                >>> global_dict.update(dataset=base_dict['.._base_.datasets.coco_detection']['dataset'])  # only update `dataset`
    """  # noqa: E501

    def __init__(self,
                 global_dict: dict,
                 base_dict: Optional[dict] = None,
                 filename: Optional[str] = None):
        self.base_dict = base_dict if base_dict is not None else {}
        self.global_dict = global_dict
        # In Windows, the filename could be like this:
        # "C:\\Users\\runneradmin\\AppData\\Local\\"
        # Although it has been an raw string, ast.parse will firstly escape
        # it as the executed code:
        # "C:\Users\runneradmin\AppData\Local\\\"
        # As you see, the `\U` will be treated as a part of
        # the escape sequence during code parsing, leading to an
        # parsing error
        # Here we use `encode('unicode_escape').decode()` for double escaping
        if isinstance(filename, str):
            filename = filename.encode('unicode_escape').decode()
        self.filename = filename
        self.imported_obj: set = set()
        super().__init__()

    def visit_ImportFrom(
        self, node: ast.ImportFrom
    ) -> Optional[Union[List[ast.Assign], ast.ImportFrom]]:
        """Hack the ``from ... import ...`` syntax and update the global_dict.

        Examples:
            >>> from mmdet.models import RetinaNet

        Will be parsed as:

        Examples:
            >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')

        ``global_dict`` will also be updated by ``base_dict`` as the
        class docstring says.

        Args:
            node (ast.AST): The node of the current import statement.

        Returns:
            Optional[List[ast.Assign]]: There three cases:

                * If the node is a statement of importing base files.
                  None will be returned.
                * If the node is a statement of importing a builtin module,
                  node will be directly returned
                * Otherwise, it will return the assignment statements of
                  ``LazyObject``.
        """
        # Built-in modules will not be parsed as LazyObject
        module = f'{node.level*"."}{node.module}'
        if _is_builtin_module(module):
            # Make sure builtin module will be added into `self.imported_obj`
            for alias in node.names:
                if alias.asname is not None:
                    self.imported_obj.add(alias.asname)
                elif alias.name == '*':
                    raise ConfigParsingError(
                        'Cannot import * from non-base config')
                else:
                    self.imported_obj.add(alias.name)
            return node

        if module in self.base_dict:
            for alias_node in node.names:
                if alias_node.name == '*':
                    self.global_dict.update(self.base_dict[module])
                    return None
                if alias_node.asname is not None:
                    base_key = alias_node.asname
                else:
                    base_key = alias_node.name
                self.global_dict[base_key] = self.base_dict[module][
                    alias_node.name]
            return None

        nodes: List[ast.Assign] = []
        for alias_node in node.names:
            # `ast.alias` has lineno attr after Python 3.10,
            if hasattr(alias_node, 'lineno'):
                lineno = alias_node.lineno
            else:
                lineno = node.lineno
            if alias_node.name == '*':
                # TODO: If users import * from a non-config module, it should
                # fallback to import the real module and raise a warning to
                # remind users the real module will be imported which will slow
                # down the parsing speed.
                raise ConfigParsingError(
                    'Illegal syntax in config! `from xxx import *` is not '
                    'allowed to appear outside the `if base:` statement')
            elif alias_node.asname is not None:
                # case1:
                # from mmengine.dataset import BaseDataset as Dataset ->
                # Dataset = LazyObject('mmengine.dataset', 'BaseDataset')
                code = f'{alias_node.asname} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.asname)
            else:
                # case2:
                # from mmengine.model import BaseModel
                # BaseModel = LazyObject('mmengine.model', 'BaseModel')
                code = f'{alias_node.name} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.name)
            try:
                nodes.append(ast.parse(code).body[0])  # type: ignore
            except Exception as e:
                raise ConfigParsingError(
                    f'Cannot import {alias_node} from {module}'
                    '1. Cannot import * from 3rd party lib in the config '
                    'file\n'
                    '2. Please check if the module is a base config which '
                    'should be added to `_base_`\n') from e
        return nodes

    def visit_Import(self, node) -> Union[ast.Assign, ast.Import]:
        """Work with ``_gather_abs_import_lazyobj`` to hack the ``import ...``
        syntax.

        Examples:
            >>> import mmcls.models
            >>> import mmcls.datasets
            >>> import mmcls

        Will be parsed as:

        Examples:
            >>> # import mmcls.models; import mmcls.datasets; import mmcls
            >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

        Args:
            node (ast.AST): The node of the current import statement.

        Returns:
            ast.Assign: If the import statement is ``import ... as ...``,
            ast.Assign will be returned, otherwise node will be directly
            returned.
        """
        # For absolute import like: `import mmdet.configs as configs`.
        # It will be parsed as:
        # configs = LazyObject('mmdet.configs')
        # For absolute import like:
        # `import mmdet.configs`
        # `import mmdet.configs.default_runtime`
        # This will be parsed as
        # mmdet = LazyObject(['mmdet.configs.default_runtime', 'mmdet.configs])
        # However, visit_Import cannot gather other import information, so
        # `_gather_abs_import_LazyObject` will gather all import information
        # from the same module and construct the LazyObject.
        alias_list = node.names
        assert len(alias_list) == 1, (
            'Illegal syntax in config! import multiple modules in one line is '
            'not supported')
        # TODO Support multiline import
        alias = alias_list[0]
        if alias.asname is not None:
            self.imported_obj.add(alias.asname)
            if _is_builtin_module(alias.name.split('.')[0]):
                return node
            return ast.parse(  # type: ignore
                f'{alias.asname} = LazyObject('
                f'"{alias.name}",'
                f'location="{self.filename}, line {node.lineno}")').body[0]
        return node

PYTHON_ROOT_DIR = osp.dirname(osp.dirname(sys.executable))

def _is_builtin_module(module_name: str) -> bool:
    """Check if a module is a built-in module.

    Arg:
        module_name: name of module.
    """
    if module_name.startswith('.'):
        return False
    if module_name.startswith('mmengine.config'):
        return True
    if module_name in sys.builtin_module_names:
        return True
    spec = find_spec(module_name.split('.')[0])
    # Module not found
    if spec is None:
        return False
    origin_path = getattr(spec, 'origin', None)
    if origin_path is None:
        return False
    origin_path = osp.abspath(origin_path)
    if ('site-package' in origin_path or 'dist-package' in origin_path
            or not origin_path.startswith(PYTHON_ROOT_DIR)):
        return False
    else:
        return True

def _gather_abs_import_lazyobj(tree: ast.Module,
                               filename: Optional[str] = None):
    """Experimental implementation of gathering absolute import information."""
    if isinstance(filename, str):
        filename = filename.encode('unicode_escape').decode()
    imported = defaultdict(list)
    abs_imported = set()
    new_body: List[ast.stmt] = []
    # module2node is used to get lineno when Python < 3.10
    module2node: dict = dict()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip converting built-in module to LazyObject
                if _is_builtin_module(alias.name):
                    new_body.append(node)
                    continue
                module = alias.name.split('.')[0]
                module2node.setdefault(module, node)
                imported[module].append(alias)
            continue
        new_body.append(node)

    for key, value in imported.items():
        names = [_value.name for _value in value]
        if hasattr(value[0], 'lineno'):
            lineno = value[0].lineno
        else:
            lineno = module2node[key].lineno
        lazy_module_assign = ast.parse(
            f'{key} = LazyObject({names}, location="{filename}, line {lineno}")'  # noqa: E501
        )  # noqa: E501
        abs_imported.add(key)
        new_body.insert(0, lazy_module_assign.body[0])
    tree.body = new_body
    return tree, abs_imported
