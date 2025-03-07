import os
import sys
sys.path.append('/home/human/codes/liyiheng/codes/posegpt')

import argparse
import torch
import torch.utils
from posegpt.utils import Config
from posegpt.datasets.posegpt import build_data_module
from torch.utils.data import DataLoader
from tqdm import tqdm
from llava import conversation as conversation_lib
from scripts.gpt_eval_full_mask import load_pretrained_model
# from posegpt.utils.vis_for_tasks import render_smpl, get_smpl_pose_params
import cv2
from posegpt.utils.vis_utils import vis_mesh
from posegpt.models.metrics import NLPEvaluator

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

device = 'cuda:0'
def eval_model(args):
    # disable_torch_init()
    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']
    
    # build model, tokenizer 
    print('Load model...')
    model, image_processor = load_pretrained_model(
        config, args.model_path, args.model_base, torch_dtype=torch_dtype, device_map={"": local_rank}, **config)

    # build dataset
    print('Load data...')
    data_module = build_data_module(eval_dataset_config=config.data.eval, image_processor=image_processor, debug=True)
    dataset = data_module['eval_dataset']
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, 
                            collate_fn=data_module['data_collator'])
    nlp_evaluator = NLPEvaluator(metric_list=['bleu', 'rouge', 'meteor'])
    print(f'&&&&&&&&&&&&&&&{config.evaluate_task}&&&&&&&&&&&&&&&&')
    for i, data in enumerate(tqdm(dataloader)):
        # if i < 1900: continue
        # move data to cuda
        data = {k: v.to(device, dtype=torch_dtype) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        imgA_path = data.pop('imgA_path', None)
        imgB_path = data.pop('imgB_path', None)
        if '3dpw' in imgB_path[0] or 'h36m' in imgB_path[0]:
            continue
        # forward
        with torch.no_grad(): output = model.evaluate(**data)

        # output result
        print(data['tasks'][0])
        print(data['caption'])
        if config.evaluate_task == 'pose2text':
            print(output['pred_text'])
            gt_pose_params = get_smpl_pose_params(output['gt_pose'], type='rotmat')
            render_smpl(gt_pose_params, save_path='gt_smpl.png')
        elif config.evaluate_task == 'text2pose':
            # forward
            with torch.no_grad(): output = model.evaluate(**data)
            # output result
            render_smpl(get_smpl_pose_params(output['gt_pose'], type='rotmat'), save_path='gt_smpl.png')
            render_smpl(get_smpl_pose_params(output['pred_pose'], type='rotmat'), save_path='pred_smpl.png', viewpoints=[[20, (0, 1, 0)]])
        elif config.evaluate_task == 'pose_difference':
            gt_pose_A, gt_pose_B = output['gt_pose']
            render_smpl(get_smpl_pose_params(gt_pose_A, type='rotmat'), save_path='gt_smpl_a.png', viewpoints=[[20, (0, 1, 0)]])
            render_smpl(get_smpl_pose_params(gt_pose_B, type='rotmat'), save_path='gt_smpl_b.png', viewpoints=[[20, (0, 1, 0)]])
            print(output['pred_text'])
        elif config.evaluate_task == 'pose_edit':            
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat'), save_path='gt_smpl_a.png', viewpoints=[[20, (0, 1, 0)]])
            # import ipdb; ipdb.set_trace()
            # data['caption'][0] = 'Put down your left leg and stretch it as straight as possible. Extend your right arm forward and your left arm backward'
            # with torch.no_grad(): output = model.evaluate(**data)
            render_smpl(get_smpl_pose_params(output['gt_pose'], type='rotmat'), save_path='gt_smpl_b.png', viewpoints=[[20, (0, 1, 0)]])
            render_smpl(get_smpl_pose_params(output['pred_pose'], type='rotmat'), save_path='pred_smpl_b.png', viewpoints=[[20, (0, 1, 0)]])
        elif config.evaluate_task == 'image2pose':
            # 右手向后伸，与身体呈45度，
            # change posegpt/utils/vis_for_tasks.py # 125 line
            # data['tasks'][0]['input'] = 'The right elbow is straight. Extend your right hand as far back as possible. The right arm is located behind the body.  The left knee partially bent while the left forearm is aligned horizontally. Can you examine the image <image> and identify the SMPL pose parameters of the individual?'
            # with torch.no_grad(): output = model.evaluate(**data) 
            vis_mesh(output['pred_axis_angles'][0].float().cpu())
            import ipdb; ipdb.set_trace()

            # cv2.imwrite('gt_img.png', cv2.imread(imgA_path[0]))
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat', normalize_root_orient=False), save_path='gt_smpl.png', viewpoints=[[-45, (0, 1, 0)]])

            render_smpl(get_smpl_pose_params(output['pred_axis_angles'].reshape(1, -1, 3), type='axis_angle', normalize_root_orient=False), save_path='pred_smpl.png', viewpoints=[[-45, (0, 1, 0)]])
        elif config.evaluate_task == 'image2text':
            # change posegpt/utils/vis_for_tasks.py # 125 line
            # i == 1905
            cv2.imwrite('gt_img.png', cv2.imread(imgA_path[0]))
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl.png', viewpoints=[[-45, (0, 1, 0)]])
            print(output['pred_text'])
            metric = nlp_evaluator(predictions=output['pred_text'], references=output['gt_text'])
            print(f"bleu: {metric['bleu']['bleu']}, rouge: {metric['rouge']['rougeL']}, meteor: {metric['meteor']['meteor']}")
        elif config.evaluate_task == 'image_difference':
            # imgA_path = 'processed_dataset/image_dataset/3dpw/imageFiles_outdoors_fencing_01_image_00917.png'
            # imgB_path = 'processed_dataset/image_dataset/3dpw/imageFiles_outdoors_fencing_01_image_00648.png'

            # imageA = cv2.cvtColor(cv2.imread(imgA_path), cv2.COLOR_BGR2RGB)
            # imageB = cv2.cvtColor(cv2.imread(imgB_path), cv2.COLOR_BGR2RGB)
            # imageA = dataloader.dataset.datasets[2].image_processor.preprocess(imageA, return_tensors='pt')['pixel_values'][0]
            # imageB = dataloader.dataset.datasets[2].image_processor.preprocess(imageB, return_tensors='pt')['pixel_values'][0]
            # data['images'] = torch.stack([imageA, imageB]).cuda().bfloat16()
            cv2.imwrite('gt_img_a.png', cv2.imread(imgA_path[0]))
            cv2.imwrite('gt_img_b.png', cv2.imread(imgB_path[0]))
            gt_pose_A, gt_pose_B = output['gt_pose']
            render_smpl(get_smpl_pose_params(gt_pose_A, type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl_a.png', viewpoints=[[-45, (0, 1, 0)]])
            render_smpl(get_smpl_pose_params(gt_pose_B, type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl_b.png', viewpoints=[[-45, (0, 1, 0)]])
            metric = nlp_evaluator(predictions=output['pred_text'], references=output['gt_text'])
            print(f"bleu: {metric['bleu']['bleu']}, rouge: {metric['rouge']['rougeL']}, meteor: {metric['meteor']['meteor']}")
            print(output['pred_text'])
        else:
            continue
        # if metric['bleu']['bleu'] > 0.15:
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default='_checkpoints/stage-finetune-full-mask-instructions-fix-hmr-bug-epoch-6/checkpoint-7260')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--config", type=str, default='configs/vis.py')
    parser.add_argument('--bf16', default=True)
    args = parser.parse_args()

    eval_model(args)
