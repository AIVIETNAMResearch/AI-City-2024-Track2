import os
import sys
import argparse
from tqdm import tqdm
from os.path import join as osp 
sys.path.append('../../utils')
from utils import phase2num, num2phase, load_json, save_json, get_question_data, get_rewrite_data     

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
    type = args.type
    root = '../../../dataset/external/BDD_PC_5K'
    root_output_dir = '../../../aux_dataset/extracted_frames/external'
    save_folder = '../../../aux_dataset/train_data/external'
    
    os.makedirs('../../../aux_dataset/train_data', exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    
    # Video frame properties
    width = 1280
    height = 720

    if type == 'all':
        types = ['train', 'val']
    else:
        types = [type]
    
    for type in types:
        print(f"Start creating data for: {type} set")
        
        anno_root = osp(root, 'annotations')
        video_root = osp(root, f'videos/{type}')
        caption_anno_root = osp(anno_root, f'caption/{type}')
        bbox_anno_root = osp(anno_root, f'bbox_annotated/{type}')
        output_dir = osp(root_output_dir, type)
        video_paths = sorted(os.listdir(video_root), key=lambda x: int(x[5:][:-4]))
        
        # Convert output_dir to abs path
        output_dir = os.path.abspath(output_dir)

        if args.choice == 'segment':
            pedes_data = {'appearance':dict(id=0, data=[]), 'environment':dict(id=0, data=[]), 'location':dict(id=0, data=[]), 'attention':dict(id=0, data=[])}
            vehicle_data = {'appearance':dict(id=0, data=[]), 'environment':dict(id=0, data=[]), 'location':dict(id=0, data=[]), 'action':dict(id=0, data=[])}
        else:
            pedes_result = {}
            for segment_type in ['appearance', 'environment', 'location', 'attention']:
                pedes_result[segment_type] = convert_dict(load_json(f"../../../aux_dataset/results/{type}/external/pedes_{segment_type}.json"))
            rewrite_pedes_data = dict(id=0, data=[])
            
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
                    phase_captions[phase_number] = dict(caption_pedestrian=e['caption_pedestrian'],
                                                        caption_vehicle=e['caption_vehicle'],
                                                        start_time=float(e['start_time']),
                                                        end_time=float(e['end_time']),
                                                        bbox=bounding_box_dict[phase_number]['bbox'],
                                                        frame_id=bounding_box_dict[phase_number]['image_id'])
            
            if args.choice == 'segment':
                try:
                    information_dict = process_information_phases(f'../../../aux_dataset/segmentation_data/mistral/{type}/external/post_processed_2/{video_name}_post_process.json')
                except:
                    continue
            else:
                if pedes_result['location'][video_name] is None:
                    continue
                
            for phase_number, phase_anno in phase_captions.items():
                if args.choice == 'rewrite':
                    if pedes_result['location'][video_name][phase_number] is None:
                        continue
                
                image_path = osp(output_dir, video_name, str(phase_anno['frame_id']) + '.png')
                (x,y,w,h) = phase_anno['bbox']
                
                # normalized x, y
                x1, y1, x2, y2 = x, y, x+w, y+h
                x1 = int((x1/width)*1000)
                y1 = int((y1/height)*1000)
                x2 = int((x2/width)*1000)
                y2 = int((y2/height)*1000)

                bbox_prompt = f"({str(x1)},{str(y1)}),({str(x2)},{str(y2)})"

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
                                        bbox_prompt=bbox_prompt,
                                        segment_type=segment_type,
                                        response=response
                                    )
                                )
                                current_data[segment_type]['id'] += 1
                else:
                    rewrite_information_pedes_dict = dict()
                    for segment_type in pedes_result.keys():               
                        rewrite_information_pedes_dict[segment_type] = pedes_result[segment_type][video_name][phase_number][segment_type]
                    
                    rewrite_pedes_data['data'].append(get_rewrite_data(id=rewrite_pedes_data['id'],
                                                                       image_path=image_path, 
                                                                       bbox_prompt=bbox_prompt, 
                                                                       data=rewrite_information_pedes_dict, 
                                                                       data_type='pedes', 
                                                                       response=phase_anno['caption_pedestrian']))
                    rewrite_pedes_data['id'] += 1

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
            else:
                # Take subset of val for validation
                save_json(path=f"{save_folder}/{type}_pedes_rewrite.json", data=rewrite_pedes_data['data'][:50])
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['train', 'val', 'all'], default='all', help='Subset choices')
    parser.add_argument("--choice", choices=['segment', 'rewrite'], default='segment', help='Segment or rewrite training')
    args = parser.parse_args()
    main(args)