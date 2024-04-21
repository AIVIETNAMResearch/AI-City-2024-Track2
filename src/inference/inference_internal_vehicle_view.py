import os
import sys
import argparse
from tqdm import tqdm
from peft import  PeftModel
from os.path import join as osp
sys.path.append('../utils')
from utils import phase2num, load_json, save_json, get_question_prompt_vehicle_view, get_rewrite_prompt_vehicle_view     
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def main(args):
    ckpt = load_json(args.ckpt_path)['internal']['vehicle_view']
    type = args.type
    root = '../../dataset'
    root_output_dir = '../../aux_dataset/extracted_frames/internal/vehicle_view'
    
    os.makedirs(f'../../aux_dataset/results/{type}', exist_ok=True)
    os.makedirs(f'../../aux_dataset/results/{type}/internal', exist_ok=True)
    os.makedirs(f'../../aux_dataset/results/{type}/internal/vehicle_view', exist_ok=True)
    
    # Video frame properties
    width = 1920
    height = 1080
    
    for data_type in ckpt.keys():
        # data_type can be: pedes, vehicle
        for segment_type in ckpt[data_type].keys():
            # segment_type can be: appearance, environment, location, attention, action, rewrite
            # Load model
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                device_map='auto',
                trust_remote_code=True,
                resume_download=True,
            )
            print(f'Loading checkpoint from {ckpt[data_type][segment_type]}')
            model = PeftModel.from_pretrained(model, ckpt[data_type][segment_type], device_map='auto').eval()
            if segment_type == 'rewrite':
                model.generation_config = GenerationConfig(temperature=0.8, top_p=0.85, num_beams=3, top_k=5, do_sample=True).from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            else:
                model.generation_config = GenerationConfig().from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

            if segment_type == 'rewrite':
                data_type_result = {}
                if data_type == 'pedes':
                    for segment_type_ in ['appearance', 'environment', 'location', 'attention']:
                        data_type_result[segment_type_] = load_json(f"../../aux_dataset/results/{type}/internal/vehicle_view/pedes_{segment_type_}.json")
                else:
                    for segment_type_ in ['appearance', 'environment', 'location', 'action']:
                        data_type_result[segment_type_] = load_json(f"../../aux_dataset/results/{type}/internal/vehicle_view/vehicle_{segment_type_}.json")

            print(f"Start inference data for vehicle view of internal dataset: {type} set - {data_type} - {segment_type}")
            anno_root = osp(root, 'annotations')
            video_roots = [osp(root, f'videos/{type}'), osp(root, f'videos/{type}/normal_trimmed')]
            caption_anno_roots = [osp(anno_root, f'caption/{type}'), osp(anno_root, f'caption/{type}/normal_trimmed')]
            bbox_anno_roots = [osp(anno_root, f'bbox_annotated', 'pedestrian', type), osp(anno_root, f'bbox_annotated', 'pedestrian', type, 'normal_trimmed')]
            output_dir = osp(root_output_dir, type)
            
            final_result = dict()
            for caption_anno_root, bbox_anno_root, video_root in zip(caption_anno_roots, bbox_anno_roots, video_roots):
                video_paths = os.listdir(video_root)
                for video_name in tqdm(video_paths):
                    try:
                        caption_anno = load_json(osp(caption_anno_root, video_name, 'vehicle_view', video_name) + '_caption.json')
                    except:
                        print(f'Error loading caption json for {video_name}')
                        continue
                    
                    try:
                        bounding_box_anno = load_json(osp(bbox_anno_root, video_name, 'vehicle_view', video_name + '_vehicle_view_bbox.json'))['annotations']
                    except:
                        print(f'Error loading bounding box json for {video_name}')
                        continue
                    
                    # Process phase bounding box
                    bounding_box_dict = dict()
                    for bb in bounding_box_anno:
                        phase_number = str(bb['phase_number'])
                        bounding_box_dict[phase_number] = bb

                    # Process phase annotation
                    phase_captions = dict()
                    for e in caption_anno['event_phase']:
                        if e['labels'][0].isdigit():
                            phase_number = str(e['labels'][0])
                        else:
                            phase_number = phase2num[e['labels'][0]]
                        
                        if bounding_box_dict.get(phase_number, False):
                            phase_captions[phase_number] = dict(start_time=float(e['start_time']),
                                                                end_time=float(e['end_time']),
                                                                bbox=bounding_box_dict[phase_number]['bbox'],
                                                                frame_id=bounding_box_dict[phase_number]['image_id'])

                    video_result = dict()
                    for phase_number, phase_anno in phase_captions.items():                    
                        image_path = osp(output_dir, video_name, str(phase_anno['frame_id']) + '.png')
                        (x,y,w,h) = phase_anno['bbox']
                        
                        # normalized x, y
                        x1, y1, x2, y2 = x, y, x+w, y+h
                        x1 = int((x1/width)*1000)
                        y1 = int((y1/height)*1000)
                        x2 = int((x2/width)*1000)
                        y2 = int((y2/height)*1000)

                        bbox_prompt = f"({str(x1)},{str(y1)}),({str(x2)},{str(y2)})"

                        if segment_type != 'rewrite':
                            sys_prompt = "Considering you are a driver and you are looking from the vehicle's third person view."
                            question_prompt = get_question_prompt_vehicle_view(image_path, bbox_prompt, segment_type, sys_prompt)
                            response, _ = model.chat(tokenizer, query=question_prompt, history=None)
                            video_result[phase_number] = {f'{segment_type}': response.strip()}
                        else:
                            rewrite_information_dict = dict()
                            for segment_type_ in data_type_result.keys():
                                rewrite_information_dict[segment_type_] = data_type_result[segment_type_][video_name][phase_number][segment_type_]
                            question_prompt = get_rewrite_prompt_vehicle_view(image_path, bbox_prompt, rewrite_information_dict, data_type)
                            response, _ = model.chat(tokenizer, query=question_prompt, history=None)
                            
                            if data_type == 'pedes':
                                video_result[phase_number] = {f'caption_pedestrian': response.strip()}
                            else:
                                video_result[phase_number] = {f'caption_vehicle': response.strip()}
                    
                    final_result[video_name] = video_result
            
            # Save to result to json file
            save_json(f'../../aux_dataset/results/{type}/internal/vehicle_view/{data_type}_{segment_type}.json', final_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['train', 'val', 'test'], default='test', help='Subset choices')
    parser.add_argument("--ckpt_path", type=str, default='./ckpt.json')
    args = parser.parse_args()
    main(args)