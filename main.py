from flask import Flask
from flask import request
from flask import render_template, redirect

from flask import session
import os
import shutil
from ultralytics import YOLO
from os import listdir
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
app = Flask(__name__,static_folder="static")
app.config['SECRET_KEY']="5b38897f6f7b7bb3fcb2c8a55027235710df24b1"
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def index():
    return render_template('main2.html')


@app.route("/submitone", methods=['POST'])
def submitone():
    file = request.files['image']
    cost= int(request.form['cost'])
    fakel=0
    door=0
    vylet=0
    rasx=0.3
    try:
        fakel= int(request.form['fakel'])
    except:
        fakel=0.1
    try:
        vylet= int(request.form['vylet'])
    except:
        vylet=0.08
    try:
        door= int(request.form['door'])
    except:
        door=0.8

    def get_conture(path, model, conf):
        m = {}
        results = model(path, conf=conf)
        img = cv2.imread(path)
        # Получение классов и имен классов
        classes = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names

        # Получение бинарных масок и их количество
        masks = results[0].masks.data  # Формат: [число масок, высота, ширина]
        num_masks = masks.shape[0]

        # Определение случайных цветов и прозрачности для каждой маски
        np.random.seed(44)
        colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]  # Случайные цвета
        print(colors)

        # Создание изображения для отображения масок
        mask_overlay = np.zeros_like(img)

        labeled_image = img.copy()

        # Добавление подписей к маскам
        for i in range(num_masks):
            color = colors[i]  # Случайный цвет
            mask = masks[i].cpu()

            # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
            mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_pixel_count = np.sum(mask_resized > 0.0)

            # Получение класса для текущей маски
            class_index = int(classes[i])
            class_name = class_names[class_index]

            m[class_name] = m.get(class_name, [])
            m[class_name].append(mask_pixel_count)

            # Добавление подписи к маске
            mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled_image, mask_contours, -1, color, thickness=cv2.FILLED)
            # cv2.putText(labeled_image, class_name, (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Отобразите итоговое изображение с наложенными масками и подписями
        # plt.figure(figsize=(8, 8), dpi=150)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(labeled_image)
        # plt.axis('off')
        # plt.show()

        k = 0
        for i in m.values():
            for j in i:
                k += j

        if "front_left_door" in m:
            return sum(m["front_left_door"]) / k, labeled_image
        elif "front_right_door" in m:
            return sum(m["front_right_door"]) / k, labeled_image
        elif "front_door" in m:
            return sum(m["front_door"]) / k, labeled_image
        else:
            return labeled_image


    filename=""
    if file:
        #os.mkdir("img/uploads/"+str(file.filename))
        filename = "static/img/uploads/"+(file.filename)
        file.save(filename)
    filename2 = "static/img/uploads/" + (file.filename)
    path =  filename
    model = YOLO("best.pt")
    try:
        coef, img=get_conture(path, model, 0.4)
        coef=float(coef)
        img = Image.fromarray(img)
        img.save(filename2)
        pth=f"<img src='{filename2}'>"
        squ=(door)*1.3*coef
        otn=(fakel+vylet)/fakel
        otn=otn*otn
        litres=squ*otn*rasx
        prices=litres*cost
        prices=int(prices)
        return render_template("resultone.html",pth=filename2,res=str(prices))
    except:
        return render_template("resultone.html",pth=filename2,res="Автомобиль не обнаружен")
app.run()