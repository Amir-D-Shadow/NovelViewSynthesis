from skimage import exposure
import cv2
import os

def match_histagram(data_path,save_path):


    for files in os.listdir(data_path):

        os.makedirs(f"{save_path}/{files}",exist_ok = True)

        os.makedirs(f"{save_path}/{files}/images",exist_ok = True)

        for img_name in os.listdir(f"{data_path}/{files}/images"):


            src = cv2.imread(f"{data_path}/{files}/images/{img_name}")

            ref = cv2.imread(f"{data_path}/{files}/images/bottomview{files}.png")

            multi = True if src.shape[-1] > 1 else False
            
            matched = exposure.match_histograms(src, ref, multichannel=multi)

            
            cv2.imwrite(f"{save_path}/{files}/images/{img_name}",matched)





if __name__ == "__main__":


    scene_name = "room1"

    src_path = f"{os.getcwd()}/data/{scene_name}"

    save_path = f"{os.getcwd()}/store"

    match_histagram(src_path,save_path)




                 

                 
