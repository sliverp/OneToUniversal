import os
import random
import cv2
import tqdm
from degrade_image_gen import gaussian_noise, motion_blur, haze, low_light, rainy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

degradation_funcs = [gaussian_noise, motion_blur, haze, low_light, rainy]
# degradation_funcs = [gaussian_noise]

source_dataset_path = '/data/lyh/train2017'
dest_dataset_path = '/data/lyh/train2017_degrade'
lock = multiprocessing.Lock()
parsed_data_record_file = open(os.path.join('.', 'images.txt'), mode='a', encoding='utf-8')
parsed_data_record = []


def start_gen_dataset(image_name):
    try:
        source_image_path = os.path.join(source_dataset_path, image_name)
        image_id, image_sufix = image_name.split('.')
        degrade_severitys = [[random.randint(0,10) for i in range(5)] for j in range(100)]
        for degrade_severity in degrade_severitys:
            image = cv2.imread(source_image_path)
            for severity, degrade_func in  zip(degrade_severity, degradation_funcs):
                image = degrade_func(image, severity / 10)
            dest_image_name = image_id + '#' + '_'.join(list(map(str, degrade_severity))) + '.' + image_sufix
            dest_iamge_path = os.path.join(dest_dataset_path, dest_image_name)
            cv2.imwrite(dest_iamge_path, image)
        return image_name + '\n'
    except:
        import traceback
        traceback.print_exc()
        return image_name + '\n'


def start():
    cur_dataset = open(os.path.join('.', 'images.txt'), mode='r', encoding='utf-8').readlines()
    cur_dataset = set(list(map(lambda x: x[:-1], cur_dataset)))
    total_source = set(os.listdir(source_dataset_path))
    continue_gen_dataset = list(total_source - cur_dataset)
    print(len(continue_gen_dataset))
    process_pool = ProcessPoolExecutor(max_workers=5)
    length = len(continue_gen_dataset)//3
    with tqdm.tqdm(total=length) as bar:
        result = []
        for image_path in continue_gen_dataset[:length]:
        # for image_path in continue_gen_dataset[:4]:
            future = process_pool.submit(start_gen_dataset, image_path)
            result.append(future)
        for future in result:
            parsed_data_record_file.write(future.result())
            bar.update(1)

    process_pool.shutdown(wait=True)
    parsed_data_record_file.close()

if __name__ == '__main__':
    start()