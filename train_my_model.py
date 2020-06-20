# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:26:13 2020

@author: WKL
"""


if __name__ == '__main__':
    
    from torch.autograd import  Variable
    import torch.optim as optim
    from tools_for_model import generateTrainSet
    from my_model import Net, weights_init
    import torch
    import time
    #用于保存生成的数据集
    import pickle
    
    start_time = time.time()
    
    PATH = './2048.pth'
    PATH_CPU = './2048_CPU.pth'
    
    model = Net()
    #加载已经训练的模型
    model.load_state_dict(torch.load(PATH))
    #参数初始化
    #model.apply(weights_init)

    if torch.cuda.is_available():
        print("gpu is available")
        model.cuda()    #将所有的模型参数移动到GPU上
    
    print("Load Time {} minutes".format(int((time.time()-start_time)/60)))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    
    #最外层循环次数
    EPOCH_1 = 20
    

    #ALL_EPOCH = EPOCH_1 * EPOCH * REPEAT
    #每一组数据的大小
    BATCH_SIZE = 1024
    #每一批训练数据的大小
    EPOCH = 100
    #每一批训练数据的重复次数
    REPEAT = 3
    
    #保存生成的数据集的起始编号
    StartFile = 81
    #文件夹的编号
    Start_File_Name = 1
    #保存的数据集的文件名格式
    #FileName = "./DataSet_{}/DataSet_{}_{}_{}.pckl"
    FileName = "/cluster/home/it_stu152/wkl_2048_DataSet/DataSet_{}_{}/DataSet_{}/DataSet_{}_{}_{}.pckl"
    
    #每隔一段时间保存一次模型
    for epoch_1 in range(EPOCH_1):
        print("\n----------------Epoch {}/{}----------------".format(epoch_1 + 1, EPOCH_1))
        print("Start Generate DataSet {} minutes".format(int((time.time()-start_time)/60)))
        #产生数据集
        # 返回值为epoch*batch_size*11*4*4
        X_train_epoch, y_train_epoch = generateTrainSet(EPOCH, BATCH_SIZE)
        
        #以二进制形式写入，记得是write byte（wb）
        #File = open(FileName.format(Start_File_Name, BATCH_SIZE, EPOCH, epoch_1 + StartFile), 'wb')
        File = open(FileName.format(BATCH_SIZE, EPOCH, 
                                    Start_File_Name, 
                                    BATCH_SIZE, EPOCH, epoch_1 + StartFile), 
                                    'wb')
        pickle.dump([X_train_epoch, y_train_epoch], File)
        File.close()
        
        X_train_epoch, y_train_epoch = Variable(X_train_epoch.cuda()),Variable(y_train_epoch.cuda())
        
        print("Start Pre-train {} minutes".format(int((time.time()-start_time)/60)))
        
        #pre-train，预训练
        for idx in range(EPOCH):
            # 训练数据为batch_size*11*4*4
            X_train = X_train_epoch[idx, :, :, :, :]
            y_train = y_train_epoch[idx, :]
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        _,pred = torch.max(outputs.data, 1)
        print('          pre_loss:%.3f          ' % (loss.item()))
        print('          pre_accuracy:%.3f          ' % (torch.sum(pred==y_train.data).double()*100/BATCH_SIZE))
        
        print("Start Train {} minutes".format(int((time.time()-start_time)/60)))
        #正式训练
        for idx in range(REPEAT * EPOCH):
            #随机生成训练批次位置
            pos = torch.randint(0, EPOCH, (1,1)).item()
            # 训练数据为batch_size*11*4*4
            X_train = X_train_epoch[pos, :, :, :, :]
            y_train = y_train_epoch[pos, :]
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        
        _,pred = torch.max(outputs.data, 1)
        print('          loss:%.3f          ' % (loss.item()))
        print('          accuracy:%.3f          ' % (torch.sum(pred==y_train.data).double()*100/BATCH_SIZE))
        
        #每隔一段时间保存一次模型
        torch.save(model.state_dict(), PATH)
        
    model.cpu()
    torch.save(model.state_dict(), PATH_CPU)
    print("\nFinish Time {} minutes\n".format(int((time.time()-start_time)/60)))