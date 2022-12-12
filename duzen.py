# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 14:55:10 2021

@author: EGE
"""
import random as rd

def grip_korona(grip, korona, tat):
    
    if korona >= 0.9 and grip <= 0.1:
        
        grip = yüzdelik(grip*10**23 + 0.5)
        rnd = rd.uniform(0,26)
        korona = 0.7 + rnd/100
        
        return grip, korona

    elif  tat ==1:
        
        korona = rd.uniform(85,95)
        korona = round(korona, 2)
        rnd = rd.uniform(5,56)
        grip = yüzdelik(grip - rnd)

        return grip/100, korona
    else:
        
        korona = korona
        grip = grip 
        
    return grip, korona

def eklem_ve_fıtık(bölge, eklem, fitik):
        
    if bölge == "sirt":
        
        fitik = rd.uniform(70, 90)
        fitik = round(fitik,2)
        eklem = rd.uniform(80,99)
        eklem = round(eklem, 2)
        return eklem/100, fitik/100
    
    else:
        fitik = fitik
        eklem = eklem
        return eklem, fitik
    
def kanserr(bilinç, kanser,sigara):
        
    if bilinç == 1 and sigara ==1:

        kanser = rd.uniform(80,90)
        kasner = round(kanser, 2)
        return kanser/100
    else:
       kanser = kanser

    return kanser

def bahar_nez(inp, bahar_nezlesi):

    if inp ==1:
        bahar_nezlesi = rd.uniform(70,95)
        bahar_nezlesi = round(bahar_nezlesi, 2)

    else:

        bahar_nezlesi = bahar_nezlesi
        pass

    return bahar_nezlesi

def enfeksiyon(enfek, kolestrol, ateş, agri, öksürük):

    if  ateş >= 37 and agri >=1 and öksürük <= 2:

        enfek == yüzdelik(enfek*10**15 +0.5)

    elif kolestrol == 1 and ateş>=37:

        enfek = yüzdelik(enfek*10**15 + 0.5)

    else:
        pass

    return enfek

def yüzdelik(girdi):

    if girdi < 0.01:
        girdi = 0.01

    elif girdi >= 0.99:
        girdi = 0.99

    else:
        girdi = girdi

    return girdi