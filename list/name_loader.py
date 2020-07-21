import os  #通过os模块调用系统命令

file_path1 = "C:\\Users\\LEO\\Google Drive\\internship\\ERFNet-CULane-PyTorch\\list\\samples"  #文件路径
file_path2 = "C:\\Users\\LEO\\Google Drive\\internship\\ERFNet-CULane-PyTorch\\list\\annotations_uint"
path_list1 = os.listdir(file_path1) #遍历整个文件夹下的文件name并返回一个列表
path_list2 = os.listdir(file_path2)

with open("C:\\Users\\LEO\\Google Drive\\internship\\ERFNet-CULane-PyTorch\\list\\train.txt", "a") as file:
    for i in range(0, len(path_list1)-100):
        file.write("/samples/" + path_list1[i] + " ")
        file.write("/annotations/" + path_list2[i] + "\n")
file.close()
with open("C:\\Users\\LEO\\Google Drive\\internship\\ERFNet-CULane-PyTorch\\list\\val.txt", "a") as file:
    for i in range(len(path_list1)-100, len(path_list1)-50):
        file.write("/samples/" + path_list1[i] + " ")
        file.write("/annotations/" + path_list2[i] + "\n")
file.close()
with open("C:\\Users\\LEO\\Google Drive\\internship\\ERFNet-CULane-PyTorch\\list\\test.txt", "a") as file:
    for i in range(len(path_list1)-50, len(path_list1)):
        file.write("/samples/" + path_list1[i] + " ")
        file.write("/annotations/" + path_list2[i] + "\n")
file.close()

# path_name = []#定义一个空列表
#
# for i in path_list:
#     path_name.append(i.split(".")[0]) #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名
#
# #path_name.sort() #排序
#
# for file_name in path_name:
#     # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
#     with open("D:/User/test/dataset.txt", "a") as file:
#         file.write(file_name + "\n")
#         print(file_name)
#     file.close()
