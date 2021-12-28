import cv2
import os

room_name = "000048"

folders = os.listdir(f"{os.getcwd()}/{room_name}")

save_path = f"{os.getcwd()}/data"

for i in range(len(folders)):

    imgs_dir = f"{os.getcwd()}/{room_name}/{folders[i]}/images"

    topview = cv2.imread(f"{imgs_dir}/topview{str(i)}.png")
    bottomview = cv2.imread(f"{imgs_dir}/bottomview{str(i)}.png")
    centerview = cv2.imread(f"{imgs_dir}/centerview{str(i)}.png")
    leftview = cv2.imread(f"{imgs_dir}/leftview{str(i)}.png")
    rightview = cv2.imread(f"{imgs_dir}/rightview{str(i)}.png")

    os.makedirs(f"{save_path}/{str(i)}")

    cv2.imwrite(f"{save_path}/{str(i)}/topview{str(i)}.png",topview)
    cv2.imwrite(f"{save_path}/{str(i)}/bottomview{str(i)}.png",bottomview)
    cv2.imwrite(f"{save_path}/{str(i)}/centerview{str(i)}.png",centerview)
    cv2.imwrite(f"{save_path}/{str(i)}/leftview{str(i)}.png",leftview)
    cv2.imwrite(f"{save_path}/{str(i)}/rightview{str(i)}.png",rightview)
