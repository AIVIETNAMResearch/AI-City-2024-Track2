import os
import sys
import argparse
from tqdm import tqdm
from os.path import join as osp 
sys.path.append('../../utils')
from utils import phase2num, load_json, save_json, get_question_data, get_rewrite_data, longest

def process_information_phases(path):
    information_phases = load_json(path)
    information_dict = dict()
    for information_phase in information_phases:
        if information_phase['labels'][0].isdigit():
            phase_number = information_phase['labels'][0]
        else:
            phase_number = phase2num[information_phase['labels'][0]]
        information_dict[phase_number] = dict(pedes=information_phase['post_process_pedes_detail_extraction'], vehicle=information_phase['post_process_vehicle_detail_extraction'])

    for phase_number in information_dict.keys():
        for key in ['pedes', 'vehicle']: 
            for k, v in information_dict[phase_number][key].items():
                information_dict[phase_number][key][k] = ' '.join(v).strip()
    return information_dict

def convert_dict(data, type):
    # Chose the longest  
    for k, v in data.items():            
        data[k] = longest(v, type)
    return data

def main(args):
    type = args.type
    root = '../../../dataset'
    root_output_dir = '../../../aux_dataset/extracted_frames/internal/overhead_view'
    save_folder = '../../../aux_dataset/train_data/internal/overhead_view'
    
    os.makedirs('../../../aux_dataset/train_data/internal', exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    
    # Video frame properties
    width = 1920
    height = 1080

    if type == 'all':
        types = ['train', 'val']
    else:
        types = [type]
    
    for type in types:
        print(f"Start creating data for: {type} set")
        
        anno_root = osp(root, 'annotations')
        video_roots = [osp(root, f'videos/{type}'), osp(root, f'videos/{type}/normal_trimmed')]
        caption_anno_roots = [osp(anno_root, f'caption/{type}'), osp(anno_root, f'caption/{type}/normal_trimmed')]
        bbox_anno_root_pedestrians = [osp(anno_root, f'bbox_annotated', 'pedestrian', type), osp(anno_root, f'bbox_annotated', 'pedestrian', type, 'normal_trimmed')]
        bbox_anno_root_vehicles = [osp(anno_root, f'bbox_annotated', 'vehicle', type), osp(anno_root, f'bbox_annotated', 'vehicle', type, 'normal_trimmed')]
        output_dir = osp(root_output_dir, type)
        
        # Convert output_dir to abs path
        output_dir = os.path.abspath(output_dir)

        if args.choice == 'segment':
            pedes_data = {'appearance':dict(id=0, data=[]), 'environment':dict(id=0, data=[]), 'location':dict(id=0, data=[]), 'attention':dict(id=0, data=[])}
            vehicle_data = {'appearance':dict(id=0, data=[]), 'environment':dict(id=0, data=[]), 'location':dict(id=0, data=[]), 'action':dict(id=0, data=[])}
        else:
            pedes_result = {}
            for segment_type in ['appearance', 'environment', 'location', 'attention']:
                pedes_result[segment_type] = convert_dict(load_json(f"../../../aux_dataset/results/{type}/internal/overhead_view/pedes_{segment_type}.json"), segment_type)
            rewrite_pedes_data = dict(id=0, data=[])
            
            vehicle_result = {}
            for segment_type in ['appearance', 'environment', 'location', 'action']:
                vehicle_result[segment_type] = convert_dict(load_json(f"../../../aux_dataset/results/{type}/internal/overhead_view/vehicle_{segment_type}.json"), segment_type)
            rewrite_vehicle_data = dict(id=0, data=[])
            
        for caption_anno_root, bbox_anno_root_pedestrian, bbox_anno_root_vehicle, video_root in zip(caption_anno_roots, bbox_anno_root_pedestrians, bbox_anno_root_vehicles, video_roots):
            video_paths = os.listdir(video_root)
            for video_name in tqdm(video_paths):
                try:
                    caption_anno = load_json(osp(caption_anno_root, video_name, 'overhead_view', video_name) + '_caption.json')
                except:
                    print(f'Error loading caption json for {video_name}')
                    continue
                
                if args.choice == 'segment':
                    try:
                        information_dict = process_information_phases(f'../../../aux_dataset/segmentation_data/mistral/{type}/internal/post_processed_2/{video_name}_post_process.json')
                    except:
                        continue
                else:
                    try:
                        if pedes_result['location'][video_name] is None:
                            continue
                    except:
                        continue

                overhead_videos = caption_anno['overhead_videos']
                for overhead_video in overhead_videos:
                    overhead_video_prefix = overhead_video[:-4]
                    try:
                        bbox_pedestrian = load_json(osp(bbox_anno_root_pedestrian, video_name, 'overhead_view', overhead_video_prefix + '_bbox.json'))['annotations']
                        bbox_vehicle = load_json(osp(bbox_anno_root_vehicle, video_name, 'overhead_view', overhead_video_prefix + '_bbox.json'))['annotations']
                    except:
                        print(f"No bounding box annotation match for {video_name}-{overhead_video_prefix}")
                        continue
                    
                    bbox_pedestrian_dict = dict()
                    for bb in bbox_pedestrian:
                        phase_number = str(bb['phase_number'])
                        bbox_pedestrian_dict[phase_number] = bb
                    
                    bbox_vehicle_dict = dict()
                    for bb in bbox_vehicle:
                        phase_number = str(bb['phase_number'])
                        bbox_vehicle_dict[phase_number] = bb

                    phase_captions = dict()
                    for e in caption_anno['event_phase']:
                        phase_number = str(e['labels'][0])
                        if bbox_pedestrian_dict.get(phase_number, False) and bbox_vehicle_dict.get(phase_number, False) and (bbox_pedestrian_dict[phase_number]['image_id'] == bbox_vehicle_dict[phase_number]['image_id']):
                            phase_captions[phase_number] = dict(start_time=float(e['start_time']),
                                                                end_time=float(e['end_time']),
                                                                caption_pedestrian=e['caption_pedestrian'],
                                                                caption_vehicle=e['caption_vehicle'],
                                                                bbox_pedestrian=bbox_pedestrian_dict[phase_number]['bbox'],
                                                                bbox_vehicle=bbox_vehicle_dict[phase_number]['bbox'],
                                                                frame_id=bbox_pedestrian_dict[phase_number]['image_id'])
                
                    for phase_number, phase_anno in phase_captions.items():
                        if args.choice == 'rewrite':
                            if pedes_result['location'][video_name][phase_number] is None:
                                continue
                        
                        image_path = osp(output_dir, video_name, overhead_video_prefix, str(phase_anno['frame_id']) + '.png')
                        if not os.path.exists(image_path):
                            print(f'Not found {image_path}')
                            continue

                        # normalized x, y
                        x, y, w, h = phase_anno['bbox_pedestrian']
                        x1, y1, x2, y2 = x, y, x+w, y+h
                        x1_p = int((x1/width)*1000)
                        y1_p = int((y1/height)*1000)
                        x2_p = int((x2/width)*1000)
                        y2_p = int((y2/height)*1000)

                        # normalized x, y
                        x, y, w, h = phase_anno['bbox_vehicle']
                        x1, y1, x2, y2 = x, y, x+w, y+h
                        x1_v = int((x1/width)*1000)
                        y1_v = int((y1/height)*1000)
                        x2_v = int((x2/width)*1000)
                        y2_v = int((y2/height)*1000)

                        bbox_prompt_pedes = f"({str(x1_p)},{str(y1_p)}),({str(x2_p)},{str(y2_p)})"
                        bbox_prompt_vehicle = f"({str(x1_v)},{str(y1_v)}),({str(x2_v)},{str(y2_v)})"

                        if args.choice == 'segment':
                            for data_type in ['pedes', 'vehicle']:
                                current_data = pedes_data if data_type == 'pedes' else vehicle_data
                                for segment_type in current_data.keys():
                                    response = information_dict[phase_number][data_type][segment_type]
                                    if response != "":
                                        current_data[segment_type]['data'].append(
                                            get_question_data(
                                                id=current_data[segment_type]['id'],
                                                image_path=image_path,
                                                bbox_prompt=(bbox_prompt_pedes, bbox_prompt_vehicle),
                                                segment_type=segment_type,
                                                response=response,
                                                view='overhead_view'
                                            )
                                        )
                                        current_data[segment_type]['id'] += 1
                        else:
                            rewrite_information_pedes_dict = dict()
                            for segment_type in pedes_result.keys():               
                                rewrite_information_pedes_dict[segment_type] = pedes_result[segment_type][video_name][phase_number][segment_type]
                                
                            rewrite_information_vehicle_dict = dict()
                            for segment_type in vehicle_result.keys():               
                                rewrite_information_vehicle_dict[segment_type] = vehicle_result[segment_type][video_name][phase_number][segment_type]
                            
                            rewrite_pedes_data['data'].append(get_rewrite_data(id=rewrite_pedes_data['id'],
                                                                                            image_path=image_path, 
                                                                                            bbox_prompt=(bbox_prompt_pedes, bbox_prompt_vehicle), 
                                                                                            data=rewrite_information_pedes_dict, 
                                                                                            data_type='pedes', 
                                                                                            response=phase_anno['caption_pedestrian'],
                                                                                            view='overhead_view'))
                            rewrite_pedes_data['id'] += 1
                            
                            rewrite_vehicle_data['data'].append(get_rewrite_data(id=rewrite_vehicle_data['id'],
                                                                                            image_path=image_path, 
                                                                                            bbox_prompt=(bbox_prompt_pedes, bbox_prompt_vehicle), 
                                                                                            data=rewrite_information_vehicle_dict, 
                                                                                            data_type='vehicle', 
                                                                                            response=phase_anno['caption_vehicle'],
                                                                                            view='overhead_view'))
                            rewrite_vehicle_data['id'] += 1
                        
            if args.choice == 'segment':
                for data_type in ['pedes', 'vehicle']:
                    current_data = pedes_data if data_type == 'pedes' else vehicle_data
                    for segment_type in current_data.keys():
                        if type == 'train':
                            save_json(path=f"{save_folder}/{type}_{data_type}_{segment_type}.json", data=current_data[segment_type]['data'])
                        else:
                            # Take subset of val for validation
                            save_json(path=f"{save_folder}/{type}_{data_type}_{segment_type}.json", data=current_data[segment_type]['data'][:50])
            else:
                if type == 'train':
                    save_json(path=f"{save_folder}/{type}_pedes_rewrite.json", data=rewrite_pedes_data['data'])
                    save_json(path=f"{save_folder}/{type}_vehicle_rewrite.json", data=rewrite_vehicle_data['data'])
                else:
                    # Take subset of val for validation
                    save_json(path=f"{save_folder}/{type}_pedes_rewrite.json", data=rewrite_pedes_data['data'][:50])
                    save_json(path=f"{save_folder}/{type}_vehicle_rewrite.json", data=rewrite_vehicle_data['data'][:50])
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['train', 'val', 'all'], default='all', help='Subset choices')
    parser.add_argument("--choice", choices=['segment', 'rewrite'], default='segment', help='Segment or rewrite training')
    args = parser.parse_args()
    main(args)