# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle

import natsort 
import struct
import pickle
import csv
import sys 
import argparse
from PIL import Image

sys.path.append(".")


from dataset_utils.colmap.pre_colmap import * 
from dataset_utils.etc_utils import getcolmapsingletechni


def convertmodel2dbfiles(path, offset=0):
    projectfolder = os.path.join(path, f"colmap_{offset:05d}")
    manualfolder = os.path.join(projectfolder, "manual")

    os.makedirs(manualfolder, exist_ok=True)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()


    with open(os.path.join(path, "cameras_parameters.txt"), "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for idx, row in enumerate(reader):
                if idx == 0: # skip the first row (header)
                    continue
                idx = idx - 1
                row = [float(c) for c in row if c.strip() != '']
                fx = row[0] # f # focal length
                fy = row[0] # f

                cx = row[1] # cu # principal point's x
                cy = row[2] # cv # principal point's y

                colmapQ = [row[5], row[6], row[7], row[8]] # [qw, qx, qy, qz] # rotation quaternion
                colmapT = [row[9], row[10], row[11]] # [tx, ty, tz] # translation
                cameraname = "cam" + str(idx).zfill(2) # cam00, cam01, ...
                focolength = fx

                principlepoint = [0,0]
                principlepoint[0] = cx 
                principlepoint[1] = cy  
                 
                imageid = str(idx+1)
                cameraid = imageid # 1, 2, ...
                pngname = cameraname + ".png"

                line =  imageid + " "

                for j in range(4):
                    line += str(colmapQ[j]) + " "
                for j in range(3):
                    line += str(colmapT[j]) + " "
                line = line + cameraid + " " + pngname + "\n" 
                # images.txt
                # 1 qw qx qy qz tx ty tz 1 cam00.png
                # 2 qw qx qy qz tx ty tz 2 cam01.png
                # ...
                empltyline = "\n"
                imagetxtlist.append(line)
                imagetxtlist.append(empltyline)

                newwidth = 2048
                newheight = 1088
                params = np.array((fx, fy, cx, cy,))

                # We can also set 0 instead of 1 since we use already undistorted Technicolor images
                camera_id = db.add_camera(1, newwidth, newheight, params) # 0: PINHOLE, 1: RADIAL_FISHEYE                                                                              # width and height

                cameraline = str(idx+1) + " " + "PINHOLE " + str(newwidth) +  " " + str(newheight) + " " + str(focolength) + " " + str(focolength)  + " " + str(cx) + " " + str(cy)  + "\n"
                # cameras.txt
                # 1 PINHOLE W H f cx cy
                cameratxtlist.append(cameraline)
                image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((colmapT[0], colmapT[1], colmapT[2])), image_id=idx+1)
                # input.db
                # save image path, camera info.(extrinsic, intrinsic), and image_id into input database
                db.commit()
                print("commited one")
    db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 


def imagecopy(video, offsetlist=[0],focalscale=1.0, fixfocal=None):
    import cv2
    import numpy as np
    import os 
    import json 
    
    pnglist = glob.glob(video + "/*.png")

    for pngpath in pnglist:
        pass 
    
    for offset in offsetlist:
        pnglist = glob.glob(video + "*_undist_" + str(offset).zfill(5)+"_*.png")
        
        targetfolder = os.path.join(video, f"colmap_{offset:05d}", "input")
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for pngpath in pnglist:
            cameraname = os.path.basename(pngpath).split("_")[3]
            newpath = os.path.join(targetfolder, "cam" + cameraname )
            shutil.copy(pngpath, newpath)
    

def checkimage(videopath):
    from PIL import Image

    import cv2
    imagelist = glob.glob(videopath + "*.png")
    for imagepath in imagelist:
        try:
            img = Image.open(imagepath) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
                print('Bad file:', imagepath) # print out the names of corrupt files
        bad_file_list=[]
        bad_count=0
        try:
            img.cv2.imread(imagepath)
            shape=img.shape # this will throw an error if the img is not read correctly
        except:
            bad_file_list.append(imagepath)
            bad_count +=1
    print(bad_file_list)
    

def fixbroken(imagepath, refimagepath):
    try:
        img = Image.open(imagepath) # open the image file
        print("start verifying", imagepath)
        img.verify() # if we already fixed it. 
        print("already fixed", imagepath)
    except :
        print('Bad file:', imagepath)
        import cv2
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(imagepath)
        
        img.load()
        img.save("tmp.png")

        savedimage = cv2.imread("tmp.png")
        mask = savedimage == 0
        refimage = cv2.imread(refimagepath)
        composed = savedimage * (1-mask) + refimage * (mask)
        cv2.imwrite(imagepath, composed)
        print("fixing done", imagepath)
        os.remove("tmp.png")


if __name__ == "__main__" :
    stride = 50 # preprocess for every 50-th frame
    framerangedict = {}
    # framerangedict["Birthday"] = [_ for _ in range(1,305,stride)]#151, 201)] # start from 1
    framerangedict["Birthday"] = [_ for _ in range(1,255,stride)]#151, 201)] # start from 1
    framerangedict["Fabien"] = [_ for _ in range(1,201,stride)]#51, 101)] # start from 1
    framerangedict["Painter"] = [_ for _ in range(0,372,stride)]#range(100, 150)] # start from 0
    framerangedict["Theater"] = [_ for _ in range(1,392,stride)]#(51, 101)] # start from 1
    framerangedict["Train"] = [_ for _ in range(1,341,stride)]#151, 201)] # start from 1
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--videopath", default="", type=str)
    # parser.add_argument("--no_gpu", action='store_true')
    # parser.add_argument("--skip_matching", action='store_true')
    # parser.add_argument("--resize", action="store_true")
    # parser.add_argument("--camera", default="OPENCV", type=str)
    # parser.add_argument("--colmap_executable", default="", type=str)
    # parser.add_argument("--magick_executable", default="", type=str)
    args = parser.parse_args()

    videopath = args.videopath

    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    srcscene = videopath.split("/")[-3]
    print("srcscene", srcscene)

    # if srcscene == "Birthday":
    #     print("check broken")
    #     fixbroken(videopath + "Birthday_undist_00012_09.png", videopath + "Birthday_undist_00013_09.png")

    #     fixbroken(videopath + "Birthday_undist_00173_09.png", videopath + "Birthday_undist_00172_09.png")
    #     fixbroken(videopath + "Birthday_undist_00255_02.png", videopath + "Birthday_undist_00254_02.png")
    
    # # copy image Technicolor/Undistorted/Birthday/Birthday_undist_00001_00.png to Technicolor/Undistorted/Birthday/colmap_00001/input/cam00.png
    # imagecopy(videopath, offsetlist=framerangedict[srcscene])
    
    # From Technicolor/Undistorted/Birthday/cameras_parameters.txt (f, cx, cy, ar, sk, qw, qx, qy, qz, tx, ty, tz),
    # Save image_id, camera extrinsic, image_path into Technicolor/Undistorted/Birthday/colmap_00001/manual/images.txt
    # Save image resolution, camera intrinsic into Technicolor/Undistorted/Birthday/colmap_00001/manual/cameras.txt
    # Save all the processed info. into Technicolor/Undistorted/Birthday/colmap_00001/input.db (COLMAP SQLite database)
    for offset in tqdm.tqdm(framerangedict[srcscene]):
        convertmodel2dbfiles(videopath, offset=offset)

    for offset in framerangedict[srcscene]:
        getcolmapsingletechni(videopath, offset=offset)

    #  rm -r colmap_* # once meet error, delete all colmap_* folders and rerun this script. 


