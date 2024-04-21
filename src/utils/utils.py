import json

phase2num = {
    "prerecognition":"0",
    "recognition":"1",
    "judgement":"2",
    "action":"3",
    "avoidance":"4"
    }

num2phase = {
    "0":"prerecognition",
    "1":"recognition",
    "2":"judgement",
    "3":"action",
    "4":"avoidance"
    }

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def longest(data, type):
    processed_data = dict()
    for video_name in data.keys():
        for phase_number in data[video_name].keys():
            if processed_data.get(phase_number, False):
                if len(processed_data[phase_number][type]) < len(data[video_name][phase_number][type]):
                    processed_data[phase_number] = {f"{type}": data[video_name][phase_number][type]}
            else:
                processed_data[phase_number] = {f"{type}": data[video_name][phase_number][type]}
    return processed_data

def fill_video(data):
    # Some video phase may have no people instance to be caption
    # So we use the result from other phases 
    non_none_positions = {int(key): data[key] for key in data if data[key] is not None}
    non_none_keys = sorted(non_none_positions.keys())

    # Function to find the closest non-None key
    def find_closest_non_none(position):
        closest_key = min(non_none_keys, key=lambda k: abs(k - position))
        return non_none_positions[closest_key]

    for key in list(data.keys()):
        if data[key] is None:
            data[key] = find_closest_non_none(int(key))

    return data

def get_question_prompt_vehicle_view(image_path, bbox_prompt, segment_type, sys_prompt):
    if segment_type == 'appearance':
        question = "Describe the pedestrian appearance including their clothing, height and age."
    elif segment_type == 'environment':
        question = "Describe the environment around including the weather conditions, traffic volume and road conditions."
    elif segment_type == 'location':
        question = "Describe the pedestrian's position relative to your vehicle, including their body position (on the left or right of your car), their body orientation (perpendicular, opposite or the same direction of your car), activity (walking or standing), and the location where they execute action on."
    elif segment_type == 'attention':
        question = "Describe the visibility of the pedestrian, evaluate whether or not the pedestrian is aware of your car and, if possible, identify their action."
    elif segment_type == 'action':
        question = "Describe the vehicle movement."
    else:
        raise f"Do not support segment type: {segment_type} !!!"
    return f"Picture 1: <img>{image_path}</img>\n{sys_prompt}<ref>{question}</ref><box>{bbox_prompt}</box>"

def get_question_prompt_overhead_view(image_path, bbox_prompt, segment_type, sys_prompt):
    bbox_prompt_pedes, bbox_prompt_vehicle = bbox_prompt
    if segment_type == 'appearance':
        question = f"<ref>Describe the pedestrian appearance including their clothing, height and age.</ref><box>{bbox_prompt_pedes}</box>"
    elif segment_type == 'environment':
        question = f"Describe the environment around <ref>the pedestrian</ref><box>{bbox_prompt_pedes}</box> and <ref>the vehicle</ref><box>{bbox_prompt_vehicle}</box> including the weather conditions, traffic volume and road conditions."
    elif segment_type == 'location':
        question = f"Describe position of <ref>the pedestrian</ref><box>{bbox_prompt_pedes}</box> relative to <ref>the vehicle</ref><box>{bbox_prompt_vehicle}</box>, including the pedestrian's body position (on the left or right of your car), body orientation (perpendicular, opposite or the same direction of your car), activity (walking or standing), and the location around the pedestrian."
    elif segment_type == 'attention':
        question = f"Describe the visibility of <ref>the pedestrian</ref><box>{bbox_prompt_pedes}</box>, evaluate whether or not the pedestrian is aware of <ref>the vehicle</ref><box>{bbox_prompt_vehicle}</box> and, if possible, identify the pedestrian's action."
    elif segment_type == 'action':
        question = f"<ref>Describe the vehicle movement.</ref><box>{bbox_prompt_vehicle}</box>"
    else:
        raise f"Do not support segment type: {segment_type} !!!"
    return f"Picture 1: <img>{image_path}</img>\n{sys_prompt}{question}"

def get_question_data(id, image_path, bbox_prompt, segment_type, response, view='vehicle_view'):
    '''
    id: data id
    image_path: path to image
    bbox_prompt: pedestrian bounding box
    segment_type: type of segment among - appearance, environment, location, attention, action
    response: ground truth response
    view: view to generate data
    '''
    if view == 'vehicle_view':
        sys_prompt = "Considering you are a driver and you are looking from the vehicle's third person view."
        question_prompt = get_question_prompt_vehicle_view(image_path, bbox_prompt, segment_type, sys_prompt)
    else:
        sys_prompt = ""
        question_prompt = get_question_prompt_overhead_view(image_path, bbox_prompt, segment_type, sys_prompt)
        
    return {"id": str(id),
            "conversations": [
                {
                    "from": "user",
                    "value": question_prompt
                },
                {
                    "from": "assistant",
                    "value": response
                }
            ]
        }
    
def get_rewrite_prompt_vehicle_view(image_path, bbox_prompt, data, data_type):
    if data_type == 'pedes':
        return f'''Picture 1: <img>{image_path}</img>
As a driver, you are viewing from a third-person perspective of your vehicle. Your task is to create a comprehensive caption that encapsulates the scene involving <ref>the pedestrian</ref><box>{bbox_prompt}</box> and your vehicle. Integrate the following elements into your description:
1. Pedestrian's Appearance: {data['appearance']}
2. Surrounding Environment: {data['environment']}
3. Relative location and distance between the pedestrian and your vehicle: {data['location']}
4. The attention, visibility, and action of the pedestrian: {data['attention']}
Your caption should provide a clear and detailed picture of the scene, combining all these elements into a cohesive narrative.'''
    else:
        return f'''Picture 1: <img>{image_path}</img>
As a driver, you are viewing from a third-person perspective of your vehicle. Your task is to create a comprehensive caption that encapsulates the scene involving <ref>the pedestrian</ref><box>{bbox_prompt}</box> and your vehicle. Integrate the following elements into your description:
1. Pedestrian's Appearance: {data['appearance']}
2. Surrounding Environment: {data['environment']}
3. Relative location and distance between the pedestrian and your vehicle: {data['location']}
4. The movement of your vehicle: {data['action']}
Your caption should provide a clear and detailed picture of the scene, combining all these elements into a cohesive narrative.'''

def get_rewrite_prompt_overhead_view(image_path, bbox_prompt, data, data_type):
    bbox_prompt_pedes, bbox_prompt_vehicle = bbox_prompt
    if data_type == 'pedes':
        return f'''Picture 1: <img>{image_path}</img>
As a driver, you are viewing from a third-person perspective of your vehicle. Your task is to create a comprehensive caption that encapsulates the scene involving <ref>the pedestrian</ref><box>{bbox_prompt_pedes}</box> and <ref>the vehicle</ref><box>{bbox_prompt_vehicle}</box>. Integrate the following elements into your description:
1. Pedestrian's Appearance: {data['appearance']}
2. Surrounding Environment: {data['environment']}
3. Relative location and distance between the pedestrian and your vehicle: {data['location']}
4. The attention, visibility, and action of the pedestrian: {data['attention']}
Your caption should provide a clear and detailed picture of the scene, combining all these elements into a cohesive narrative.'''
    else:
        return f'''Picture 1: <img>{image_path}</img>
As a driver, you are viewing from a third-person perspective of your vehicle. Your task is to create a comprehensive caption that encapsulates the scene involving <ref>the pedestrian</ref><box>{bbox_prompt_pedes}</box> and <ref>the vehicle</ref><box>{bbox_prompt_vehicle}</box>. Integrate the following elements into your description:
1. Pedestrian's Appearance: {data['appearance']}
2. Surrounding Environment: {data['environment']}
3. Relative location and distance between the pedestrian and your vehicle: {data['location']}
4. The movement of your vehicle: {data['action']}
Your caption should provide a clear and detailed picture of the scene, combining all these elements into a cohesive narrative.'''

def get_rewrite_data(id, image_path, bbox_prompt, data, data_type, response, view='vehicle_view'):
    if view == 'vehicle_view':
        rewrite_prompt = get_rewrite_prompt_vehicle_view(image_path, bbox_prompt, data, data_type)
    else:
        rewrite_prompt = get_rewrite_prompt_overhead_view(image_path, bbox_prompt, data, data_type)
    
    return {"id": str(id),
            "conversations": [
                {
                    "from": "user",
                    "value": rewrite_prompt
                },
                {
                    "from": "assistant",
                    "value": response
                }
            ]
        }
    
    
