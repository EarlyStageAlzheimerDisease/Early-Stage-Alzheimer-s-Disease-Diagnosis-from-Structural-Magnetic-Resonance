import os
from numpy import matlib
from tqdm import tqdm
from DMO import DMO
from DPOA import DPOA
from IDPOA import IDPOA
from LOA import LOA
from Model_3D_EfficientNet import Model_3D_EfficientNet
from Model_Resnet import Model_RESNET
from NRO import NRO
from Plot_Results import *
from Obj_Fun import Obj_fun_CLS
from Model_CNN_ import Model_CNN
from Global_Vars import Global_Vars
from keras.src.utils import to_categorical
from Model_UNET import Model_UNET

no_of_Dataset = 2


def Read_Image(filename):  # Read image files
    images = cv.imread(filename)
    Image = cv.resize(images, (512, 512)).astype('uint8')
    return Image


def Read_dataset_F():
    Directory = './Dataset/Dataset 1/'
    List_dir = os.listdir(Directory)
    Image = []
    Target = []
    for n in range(len(List_dir)):
        In_Fold = Directory + List_dir[n]
        List_In_Fold = os.listdir(In_Fold)
        for i in range(len(List_In_Fold)):
            SubFold = In_Fold + '/' + List_In_Fold[i]
            listSub = os.listdir(SubFold)
            for j in range(len(listSub)):
                Fold = SubFold + '/' + listSub[j]
                listFold = os.listdir(Fold)
                for k in range(len(listFold)):
                    filename = Fold + '/' + listFold[k]
                    Image.append(Read_Image(filename))
                    Target.append(j)
                    Tar = to_categorical(Target)
    return Image, Tar


def Read_dataset_S():
    Directory = './Dataset/Dataset 2/'
    List_dir = os.listdir(Directory)
    Images = []
    Targets = []
    for n in range(len(List_dir)):
        In_Fold = Directory + List_dir[n]
        List_In_Fold = os.listdir(In_Fold)
        List_In_Fold.remove('Alzheimer.ipynb')
        List_In_Fold.remove('OriginalDataset')
        for i in range(len(List_In_Fold)):
            SubFold = In_Fold + '/' + List_In_Fold[i]
            listSub = os.listdir(SubFold)
            for j in range(len(listSub)):
                Fold = SubFold + '/' + listSub[j]
                listFold = os.listdir(Fold)
                for k in range(len(listFold)):
                    filename = Fold + '/' + listFold[k]
                    Images.append(Read_Image(filename))
                    Targets.append(j)
                    Tars = to_categorical(Targets)
    return Images, Tars


# Read_Dataset
an = 0
if an == 1:
    Image, Tar = Read_dataset_F()
    Images, Tars = Read_dataset_S()
    np.save('Images_1.npy', Image)
    np.save('Images_2.npy', Images)
    np.save('Target_1.npy', Tar)
    np.save('Target_2.npy', Tars)


# pre-Processing
an = 0
if an == 1:
    PreProcess = []
    patch_size = 16
    for n in range(no_of_Dataset):
        image = np.load('Images_' + str(n + 1) + '.npy')

        # Check if the image was loaded properly
        if image is None:
            print(f"Error: Unable to load image from Images_{n + 1}.npy")
            continue

        for i in tqdm(range(len(image))):
            images = image[i]
            image_bw = cv.cvtColor(images, cv.COLOR_BGR2GRAY)

            clahe = cv.createCLAHE(clipLimit=5)  # The declaration of CLAHE
            final_img = clahe.apply(image_bw) + 30
            blurred_image = cv.medianBlur(final_img, ksize=5)

            # Ordinary thresholding the same image
            _, ordinary_img = cv.threshold(blurred_image, 155, 255, cv.THRESH_BINARY)

            # image patch splitting
            img_height, img_width = blurred_image.shape
            patches = []
            for y in range(0, img_height, patch_size):
                for x in range(0, img_width, patch_size):
                    patch = blurred_image[y:y + patch_size, x:x + patch_size]
                    patches.append(patch)
            PreProcess.append(patches)
        np.save('Preprocessed_images_' + str(n + 1) + '.npy', PreProcess)


# Optimization for Classification
an = 0
if an == 1:
    Fit = []
    for n in range(no_of_Dataset):
        Images = np.load('Preprocessed_images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the images
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the images
        Global_Vars.Feat = Images
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # 1 for Hidden neuron count in EfficientNet, and 1 for Epoch count in EfficientNet, 1 for Activation  in EfficientNet
        xmin = matlib.repmat([5, 5, 1], Npop, 1)
        xmax = matlib.repmat([255, 50, 5], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun_CLS
        Max_iter = 50

        print("DMO...")
        [bestfit1, fitness1, bestsol1, time1] = DMO(initsol, fname, xmin, xmax, Max_iter)  # Dwarf Mongoose Optimization

        print("LOA...")
        [bestfit2, fitness2, bestsol2, time2] = LOA(initsol, fname, xmin, xmax, Max_iter)  # Lyrebird Optimization Algorithm

        print("NRO...")
        [bestfit3, fitness3, bestsol3, time3] = NRO(initsol, fname, xmin, xmax, Max_iter)  # Nuclear Reaction Optimization

        print("DPOA...")
        [bestfit4, fitness4, bestsol4, time4] = DPOA(initsol, fname, xmin, xmax, Max_iter)  # Doctor and Patient Optimization Algorithm

        print("IDPOA...")
        [bestfit5, fitness5, bestsol5, time5] = IDPOA(initsol, fname, xmin, xmax, Max_iter)  # Improved Doctor and Patient Optimization

        Bestsol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Fitness = [fitness1, fitness1, fitness1, fitness1, fitness1]
        Fit.apppend(Fitness)
        np.save('BestSol_' + str(n + 1) + '.npy', Bestsol)  # Save the Bestsoluton
    np.save('Fitness.npy', Fit)  # Save the Fitness

# classification
an = 0
if an == 1:
    Evaluate = []
    for n in range(no_of_Dataset):
        Image = np.load('Preprocessed_images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Bestsol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Act = []
        Batch_size = [16, 32, 64, 128, 256]
        for act in range(len(Batch_size)):
            learnperc = round(Image.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Image[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Image[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 14))
            for j in range(Bestsol.shape[0]):
                print(act, j)
                sol = np.round(Bestsol[j, :]).astype(np.int16)
                Eval[n, :], pred = Model_3D_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act], sol)
            Eval[5, :], pred = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])
            Eval[6, :], pred1 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])
            Eval[7, :], pred2 = Model_UNET(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])
            Eval[8, :], pred3 = Model_3D_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])
            Eval[9, :], pred4 = Eval[4, :]  # Model Proposed
        Evaluate.append(Eval)
    np.save('Eval_all.npy', Evaluate)  # Save Eval all


plotConvResults()
Plot_ROC_Curve()
plot_results_Batch()
plot_results_Epoch()