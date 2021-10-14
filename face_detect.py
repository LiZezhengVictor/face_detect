from json import load
import os
import tkinter
import cv2
import numpy as np
from six import with_metaclass
from tensorflow.python.eager.function import register
import utils.utils as utils
from net.inception import InceptionResNetV1
from net.mtcnn import mtcnn
from time import sleep
from tkinter import *
from tkinter import messagebox,filedialog,dialog,simpledialog
import tkinter.font as tkfont
import matplotlib.image as img
from PIL import Image,ImageTk

def getFaceInfo():
    '''
        录入人脸信息的函数；
        摄像头拍照；
        获取人脸的面部信息faceInfo和名字name；
        返回名字name，面部信息faceInfo和拍摄的图片originimg
    '''

    #从文本框获取用户姓名
    #如果为空
    ID=label3["text"]
    
    #控制摄像头拍照
    # video_capture = cv2.VideoCapture(0)
    # ret, originimg = video_capture.read()
    originimg = cv2.imread(file_path)

    #对拍摄到的图片进行处理
    img = cv2.cvtColor(originimg,cv2.COLOR_BGR2RGB)
    height,width,_ = np.shape(img)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #mtcnn检测人脸
    rectangles = mtcnn_model.detectFace(img_rgb, threshold)

    #判断个数，排除不符合的
    if len(rectangles)==0:
        print("No Face Detected!")
        return 0
    if len(rectangles)>=2:
        print("Exists More Than One Face!")
        return 0
    #转化成正方形
    rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)


    #对检测到的人脸进行编码
    for rectangle in rectangles:
        # 截取图像
        landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
        crop_img = img_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        # 利用人脸关键点进行人脸对齐
        crop_img,_ = utils.Alignment_1(crop_img,landmark)
        crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
        # 编码形成特征
        face_encoding = utils.calc_128_vec(facenet_model, crop_img)
        face_encoding = list(face_encoding)

    
    return ID,face_encoding,originimg

def searchFaceInfo():
    '''
        搜索人脸信息的函数；
        摄像头拍照、处理；
        返回人脸编码后信息face_encoding
    '''
    #控制摄像头拍照
    # video_capture = cv2.VideoCapture(0)
    # ret, originimg = video_capture.read()
    originimg = cv2.imread(file_path)

    #对拍摄到的图片进行处理
    img = cv2.cvtColor(originimg,cv2.COLOR_BGR2RGB)
    height,width,_ = np.shape(img)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #mtcnn检测人脸
    rectangles = mtcnn_model.detectFace(img_rgb, threshold)

    #判断个数，排除不符合的
    if len(rectangles)==0:
        print("No Face Detected!")
        return 0
    if len(rectangles)>=2:
        print("Exists More Than One Face!")
        return 0
    #转化成正方形
    rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

    #对检测到的人脸进行编码
    for rectangle in rectangles:
        # 截取图像
        landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
        crop_img = img_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        # 利用人脸关键点进行人脸对齐
        crop_img,_ = utils.Alignment_1(crop_img,landmark)
        crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
        # 编码形成特征
        face_encoding = utils.calc_128_vec(facenet_model, crop_img)
        face_encoding = list(face_encoding)
    
    return face_encoding

def writeFaceInfo(name,faceinfo,img):
    '''
        录入人脸信息的函数；
        将人名，经过getFaceInfo()函数处理后的姓名和面部编码信息；
        保存到数据库faceinfodb.txt中的函数
    '''
    facefile = open('faceinfodb.txt','a')
    facefile.write(name+',')
    faceinfo = str(faceinfo).replace('[','').replace(']','')
    facefile.writelines(faceinfo+',\n')
    facefile.close()
    cv2.imwrite('face_dataset/' + name + '.jpg',img)

def compareFaceInfo(faceinfo):
    '''
        将传入函数的面部编码信息与数据库对比；
        返回数据库中是否有此信息；
        以及该面部对应的人名
    '''
    idList = []
    faceinfoList = []
    f = open('faceinfodb.txt','r')
    for line in f.readlines():
        data = line.split(',')
        ID = data[0]
        nowfaceinfo = list(map(float,data[1:-1]))
        idList.append(ID)
        faceinfoList.append(nowfaceinfo)
    
    matches = utils.compare_faces(faceinfoList,faceinfo)
    face_distances = utils.face_distance(faceinfoList,faceinfo)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return [True,idList[best_match_index]]   #存在这个人的脸部信息
    else:
        return [False,idList[best_match_index]]  #不存在


if __name__ == "__main__":
    #加载mtcnn模型
    mtcnn_model = mtcnn()
    threshold = [0.5,0.6,0.8]
    #加载facenet模型
    facenet_model = InceptionResNetV1()
    model_path = './model_data/facenet_keras.h5'
    file_dir='.\\test_dataset\\archive\\test_data'
    facenet_model.load_weights(model_path)

    def photo_resize(w, h, w_box, h_box, pil_image):
        f1 = 1.0*w_box/w 
        f2 = 1.0*h_box/h  
        factor = min([f1, f2])  
        width = int(w*factor)  
        height = int(h*factor)  
        resized_photo=pil_image.resize((width, height), Image.ANTIALIAS)
        return resized_photo

    global top
    top=Tk()
    w_box=500
    h_box=500
    top.title('Face_Detection')
    top.geometry('1600x800')
    m_str_var=tkinter.StringVar()
    myfont=tkfont.Font(family="Helvetica",size=12, weight="bold",slant= "italic")

    #***************************background******************************
    back1=Image.open('back.jpg')
    back=ImageTk.PhotoImage(back1)
    background=Label(top,image=back,compound=CENTER)
    background.pack()
    #***************************background******************************

    #***************************load photo******************************
    img_file1=PhotoImage(file='')
    lbl_img1=Label (top,image=img_file1,height=500,width=500,relief=SUNKEN)
    lbl_img1.place(x=0,y=0,anchor='nw')
    #***************************load photo******************************

    #***************************read photo******************************
    img_file2=PhotoImage(file='')
    lbl_img2=Label(top,image=img_file2,height=500,width=500,relief=SUNKEN)
    lbl_img2.place(x=1100,y=0,anchor='nw')
    #***************************read photo******************************

    #***************************label 1******************************
    label1=Label(top,justify=CENTER,text='',width=50,font=myfont)
    label1.place(x=550,y=550)
    #***************************label 1******************************

    #***************************label 2******************************
    label2=Label(top,justify=CENTER,text='your name is ',font=myfont)
    label2.place(x=575,y=400)
    #***************************label 2******************************

    #***************************label 3******************************
    label3=Label(top,justify=CENTER,text='',font=myfont)
    label3.place(x=700,y=400)
    #***************************label 3******************************

    #************************click button1**************************
    #获取Face
    def register():
        if(label3['text']==''):
            label1['text']="Please input your name!"
            return 0
        else:
            name,faceInfo,img = getFaceInfo()
            existFlag = compareFaceInfo(faceInfo)
            #写入后台数据库
            if(existFlag[0]):
                label1['text']='Face information already exists!'
            else:
                writeFaceInfo(name,faceInfo,img)
                label1['text']=('Successfully write face information!')
    #************************click button1**************************

    #************************click button2**************************
    def login():
        global img_file2
        label1['text']=("Please load a picture")
        faceInfo = searchFaceInfo()
        result = compareFaceInfo(faceInfo)
        print(result)
        if(result[0]):
            img_photo2=Image.open('face_dataset/'+result[1]+'.jpg')
            w,h=img_photo2.size
            pil_image_resized = photo_resize(w,h,w_box,h_box,img_photo2)
            img_file2=ImageTk.PhotoImage(pil_image_resized)
            lbl_img2.configure(image=img_file2)
            label1['text']=("Welcome "+result[1]+'!')
        else:
            '''img_photo2=Image.open('face_dataset/'+result[1]+'.jpg')
            w,h=img_photo2.size
            pil_image_resized = photo_resize(w,h,w_box,h_box,img_photo2)
            img_file2=ImageTk.PhotoImage(pil_image_resized)
            lbl_img2.configure(image=img_file2)'''
            label1['text']=('Your face information unfound!') 
    #************************click button2**************************

    #************************click button3**************************
    def load():
        global img_file1,w_box,h_box,file_path
        file_path=filedialog.askopenfilename(initialdir='')
        img_photo=Image.open(file_path)
        w,h=img_photo.size
        pil_image_resized = photo_resize(w,h,w_box,h_box,img_photo)
        img_file1=ImageTk.PhotoImage(pil_image_resized)
        lbl_img1.configure(image=img_file1)
    #************************click button3**************************

    #************************click button4**************************
    def name():
        n=simpledialog.askstring('name register','Please enter the name!')
        if n:
            label3['text']=n
    #************************click button4**************************

    #************************click button5**************************
    def autoload():
        global img_file1,w_box,h_box,file_path
        for root,dirs,files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    print(file)
                    label3['text']=os.path.splitext(file)[0]
                    file_path=os.path.join(root, file)
                    img_photo=Image.open(file_path)
                    w,h=img_photo.size
                    pil_image_resized = photo_resize(w,h,w_box,h_box,img_photo)
                    img_file1=ImageTk.PhotoImage(pil_image_resized)
                    lbl_img1.configure(image=img_file1)
                    register()
    #************************click button5**************************

    button1=Button(top,text='register',width=10,height=2,command=register)
    button1.place(x=400,y=600)
    button1.config(font=myfont)
    button2=Button(top,text='login',width=10,height=2,command=login)
    button2.place(x=1100,y=600)
    button2.config(font=myfont)
    button3=Button(top,text='load',width=10,height=2,command=load)
    button3.config(font=myfont)
    button3.place(x=400,y=700)
    button4=Button(top,text='name',width=10,height=2,command=name)
    button4.config(font=myfont)
    button4.place(x=1100,y=700)
    button5=Button(top,text='autoload',width=10,height=2,command=autoload)
    button5.config(font=myfont)
    button5.place(x=750,y=700)
    top.mainloop()