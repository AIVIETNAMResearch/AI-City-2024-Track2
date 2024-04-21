import os
import sys
import argparse
from tqdm import tqdm
from peft import  PeftModel
from os.path import join as osp
sys.path.append('../utils')
from utils import phase2num, load_json, save_json, get_question_prompt_vehicle_view, get_rewrite_prompt_vehicle_view     
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def convert_dict(data):
    for k, v in data.items():
        new_v = dict()
        if v is not None:
            for v_k, v_v in v.items():
                if not v_k.isdigit():
                    new_v[phase2num[v_k]] = v_v
                else:
                    new_v[v_k] = v_v
            data[k] = new_v
        else:
            data[k] = v
    return data

def main(args):
    ckpt = load_json(args.ckpt_path)['external']
    type = args.type
    root = '../../dataset/external/BDD_PC_5K'
    root_output_dir = '../../aux_dataset/extracted_frames/external'
    
    os.makedirs(f'../../aux_dataset/results/{type}', exist_ok=True)
    os.makedirs(f'../../aux_dataset/results/{type}/external', exist_ok=True)
    
    # Video frame properties
    width = 1280
    height = 720
    
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
                        data_type_result[segment_type_] = convert_dict(load_json(f"../../aux_dataset/results/{type}/external/pedes_{segment_type_}.json"))
                else:
                    for segment_type_ in ['appearance', 'environment', 'location', 'action']:
                        data_type_result[segment_type_] = convert_dict(load_json(f"../../aux_dataset/results/{type}/external/vehicle_{segment_type_}.json"))

            print(f"Start inference data for external dataset: {type} set - {data_type} - {segment_type}")
            anno_root = osp(root, 'annotations')
            video_root = osp(root, f'videos/{type}')
            caption_anno_root = osp(anno_root, f'caption/{type}')
            bbox_anno_root = osp(anno_root, f'bbox_annotated/{type}')
            output_dir = osp(root_output_dir, type)
            video_paths = sorted(os.listdir(video_root), key=lambda x: int(x[5:][:-4]))

            final_result = dict()
            for video_path in tqdm(video_paths):
                video_name = video_path[:-4]
                caption_anno = load_json(osp(caption_anno_root, video_name) + '_caption.json')
                bounding_box_anno = load_json(osp(bbox_anno_root, video_name) + '_bbox.json')['annotations']

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
            save_json(f'../../aux_dataset/results/{type}/external/{data_type}_{segment_type}.json', final_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['train', 'val', 'test'], default='test', help='Subset choices')
    parser.add_argument("--ckpt_path", type=str, default='./ckpt.json')
    args = parser.parse_args()
    main(args)