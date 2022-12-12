#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from duzen import grip_korona, eklem_ve_fıtık, kanserr, yüzdelik, bahar_nez, enfeksiyon

def predict_desease():
    bolge_dict = {
        '0': 0,
        'bacak': 0,
        'bel': 0,
        'genel': 0,
        'bas': 0,
        'sirt': 0
    }

    sector_dict = {
        'emekli': 0,
        'hizmet': 0,
        'tarım': 0,
        'ogrenci': 0,
        'isci': 0
    }

    output_read = open('questions.json')
    answers = json.load(output_read)

    Yas = float(answers['yas'])
    Burun_akintisi = float(answers['burun'])
    Ates = float(answers['sicaklik'])
    Agri_siddeti = float(answers['agri'])
    girdi1 = answers['bolge']
    oksuruk = float(answers['oksuruk'])
    Geniz_akin = float(answers['geniz_akinti'])
    Goz_kizarikl = float(answers['goz'])
    Bilinç_kaybi = float(answers['bilinc'])
    Kolestrol = float(answers['kalestrol'])
    seker = float(answers['seker_hasta'])
    Tat_Ve_koku_kaybi = float(answers['tat_koku'])
    Sigara = float(answers['sigara'])
    Cinsiyet = float(answers['cinsyet'])
    Alkol = int(answers['alkol'])
    girdi2 = answers['sector']

    # bu iki değer vücut kitle endeski hesabında kullanılıyor ve kitle değişkenine atanıyor
    Kilo = answers['kilogram']
    Boy = float(answers['boy']) / 100
    Kitle = float(Kilo)/(float(Boy))**2

    bolge_dict[girdi1] = 1
    sector_dict[girdi2] = 1

    a = bolge_dict['bacak']
    b = bolge_dict['bel']
    c = bolge_dict['genel']
    d = bolge_dict['bas']
    e = bolge_dict['sirt']
    g = bolge_dict['0']

    el = sector_dict['emekli']
    f = sector_dict['hizmet']
    h = sector_dict['tarım']
    j = sector_dict['ogrenci']
    k = sector_dict['isci']

    model = load_model('model3.h5')

    output = model.predict([[Yas, Burun_akintisi, Ates, Agri_siddeti, 
                a,b,c,d,e,g , oksuruk, Geniz_akin, Goz_kizarikl, Bilinç_kaybi,
                Kolestrol,seker, Tat_Ve_koku_kaybi, Sigara, Cinsiyet, 
                Alkol, Kitle, el,f,h,j,k]])

    Hastaliklar = ['Bahar Nezlesi',' Fıtık BAşlangıcı', 'Eklem ve Kas HAstalıkları', 'Enfeksiyon Hastalıkları',
    'Korona Virüs Salgını', 'Kanser veya Türevi Hastalıklar', 'Grip', 'Yüksek Kilo']
    #düzeltme fonksiyonlarına verilerin hazırlanması
    output = output[0]

    bahar = yüzdelik(output[0])
    fitik = yüzdelik(output[1])
    eklem = yüzdelik(output[2])
    enfek = yüzdelik(output[3])
    grip = yüzdelik(output[4])
    kanser = yüzdelik(output[5])
    korona = yüzdelik(output[6])
    yüksek = yüzdelik(output[7])

    # grip, korona = grip_korona(grip, korona, Tat_Ve_koku_kaybi)
    # eklem, fitik = eklem_ve_fıtık(girdi1, eklem, fitik)
    # kanserr = kanser(Bilinç_kaybi, kanserr)

    #düzeltme fonksiyonları (filtreleme işlemleri)
    grip, korona = grip_korona(grip, korona,Tat_Ve_koku_kaybi)
    eklem, fitik = eklem_ve_fıtık(girdi1, eklem, fitik)
    kanser = kanserr(Bilinç_kaybi, kanser, Sigara)
    bahar = bahar_nez(Goz_kizarikl, bahar)
    enfek = enfeksiyon(enfek, Kolestrol, Ates, Agri_siddeti, oksuruk)
    veriler = [bahar/100, fitik, eklem, enfek, grip, kanser, korona/100, yüksek]

    # ön işlemler için datafrme dönüşümü
    output_df = pd.DataFrame(output)
    output_df['Hastalıklar'] = Hastaliklar
    # print(output_df)
    print(veriler)

    return veriler

if __name__ == '__main__':
    predict_desease()