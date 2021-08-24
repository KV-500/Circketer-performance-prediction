import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas  as pd
import numpy as np
import tkinter as tk
import csv
from tkinter import messagebox
from tkinter import *
import cv2
import tkinter.ttk as ttk

def cir():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        print(dataset)

        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        print(X)
        print(Y)


        #BF vs Runs
        fig=plt.figure(figsize=(263,15))
        plt.plot('Runs',color='red')
        plt.plot('BF',color='green')
        plt.legend(['Runs','BF'],loc='best',fontsize=20)
        plt.title('BF vs Runs')
        plt.plot(dataset['Runs'])
        plt.plot(dataset['BF'])
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('Runs',fontsize=30)
        plt.show()

        #fours
        fig=plt.figure(figsize=(263,15))
        plt.plot('Runs',color='red')
        plt.plot('BF',color='green')
        plt.legend(['Runs','BF'],loc='best',fontsize=20)
        plt.title('Fours')
        plt.plot(dataset['fours'])
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('Fours',fontsize=30)
        plt.show()

        #BF vs fours
        fig=plt.figure(figsize=(263,15))
        plt.plot('fours',color='red')
        plt.plot('BF',color='green')
        plt.legend(['fours','BF'],loc='best',fontsize=20)
        plt.title('number of fours')
        plt.plot(dataset['fours'],color='red')
        plt.plot(dataset['BF'],color='green')
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('fours',fontsize=30)
        plt.show()

        #six
        fig=plt.figure(figsize=(263,15))
        plt.plot('Runs',color='red')
        plt.plot('BF',color='green')
        plt.legend(['Runs','BF'],loc='best',fontsize=20)
        plt.title('six')
        plt.plot(dataset['six'])
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('Six',fontsize=30)
        plt.show()


        #BF vs six
        fig=plt.figure(figsize=(263,15))
        plt.plot('six',color='red')
        plt.plot('BF',color='green')
        plt.legend(['six','BF'],loc='best',fontsize=20)
        plt.title('number of six')
        plt.plot(dataset['six'],color='red')
        plt.plot(dataset['BF'],color='green')
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('six',fontsize=30)
        plt.show()







        fig=plt.figure(figsize=(263,15))
        plt.title('PIE')
        FOURS=sum(dataset.fours)
        SIX=sum(dataset.six)
        RUNS=(sum(dataset.Runs)-(FOURS+SIX))
        values=[FOURS,SIX,RUNS]
        label=["Fours","Six","Runs"]
        plt.pie(values,labels=label)
        plt.show()


        #Runs
        range=[10,20,30,40,50,60,70,80,90,100]
        fig=plt.hist(dataset.Runs,range,histtype='bar',color='darkslategrey',rwidth=0.5)
        plt.xlabel("Range of Runs",size=30)
        plt.ylabel("No. of times",size=30)
        plt.title("Runs")
        plt.show()

        plt.figure(figsize=(20,10))
        plt.barh(dataset.fours,dataset.six)
        plt.title("fours & six",size=30)
        plt.show()


        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)                                              
        from sklearn.linear_model import LinearRegression
        logmodel=LinearRegression()
        logmodel.fit(X_train,Y_train)
        pred=logmodel.predict(X_test)
        print(X_test)
        print(pred)
        print(pd.DataFrame(Y_test,pred))

        from sklearn.metrics import mean_squared_error
        print(mean_squared_error(Y_test,pred))
        print(logmodel.score(X_test,Y_test))

def dataset():
        root = Tk()
        root.title("Circketer Dataset")
        width = 1200
        height = 900
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        root.geometry("%dx%d+%d+%d" % (width, height, x, y))
        root.resizable(0, 0)


        TableMargin = Frame(root, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=("Runs", "Mins", "Ball Faced","Fours","six","Postion","Dismissal","Ground","Result","Inns","SR"), height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        tree.heading('Runs', text="Runs", anchor=W)
        tree.heading('Mins', text="Mins", anchor=W)
        tree.heading('Ball Faced', text="Ball Faced", anchor=W)
        tree.heading('Fours', text="Fours", anchor=W)
        tree.heading('six', text="six", anchor=W)
        tree.heading('Postion', text="Postion", anchor=W)
        tree.heading('Dismissal', text="Dismissal", anchor=W)
        tree.heading('Ground', text="Ground", anchor=W)
        tree.heading('Result', text="Result", anchor=W)
        tree.heading('Inns', text="Inns", anchor=W)
        tree.heading('Dismissal', text="Dismissal", anchor=W)
        tree.column('#0', stretch=NO, minwidth=0, width=0)
        tree.column('#1', stretch=NO, minwidth=0, width=100)
        tree.column('#2', stretch=NO, minwidth=0, width=100)
        tree.column('#3', stretch=NO, minwidth=0, width=100)
        tree.column('#4', stretch=NO, minwidth=0, width=100)
        tree.column('#5', stretch=NO, minwidth=0, width=100)
        tree.column('#6', stretch=NO, minwidth=0, width=100)
        tree.column('#7', stretch=NO, minwidth=0, width=100)
        tree.column('#8', stretch=NO, minwidth=0, width=100)
        tree.column('#9', stretch=NO, minwidth=0, width=100)
        tree.column('#10', stretch=NO, minwidth=0, width=100)
        tree.pack()

        with open('data.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                firstname = row['Runs']
                lastname = row['Mins']
                address = row['BF']
                fours  = row['fours']
                six  = row['six']
                Pos  = row['Pos']
                Dismissal  = row['Dismissal']
                Ground  = row['Ground']
                Result  = row['Result']
                Inns  = row['Inns']
                SR = row['SR']
                tree.insert("", 0, values=(firstname, lastname, address,fours,six,Pos,Dismissal,Result,Inns,SR))

        #============================INITIALIZATION==============================
        if __name__ == '__main__':
            root.mainloop()

def alter():
        window =tk.Tk()
        window.title("INTERNSHIP")
        window.configure(background='white')
        window.geometry('1500x1000')


        x_cord = 75;
        y_cord = 20;
        message = tk.Label(window, text="" ,bg="white"  ,fg="dodger blue"  ,width=60  ,height=15, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
        message.place(x=250-x_cord, y=70-y_cord)

        dataset = pd.read_csv('data.csv')
                #print(type(dataset))
                #dataset.info()
                #sns.heatmap(dataset.isnull())
                #plt.show()
        def impute_sr(cols):
            SR=cols[0]
            BF=cols[1]
            if pd.isnull(SR):
                if BF==0:
                    return 0
                elif BF>=25:
                    return 50
                else:
                    return 30
            else:
                    return SR
            dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
                #sns.heatmap(dataset.isnull())
                #plt.show()

                #print(dataset['SR'])
        def impute_mins(cols):
            Mins=cols[0]
            BF=cols[1]

            if pd.isnull(Mins):
                if BF==0:
                    return 0
                elif BF==50:
                    return 50
                else:
                    return 100
            else:
                return Mins
            dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
            Runs=cols[0]
            fours=cols[1]
            if pd.isnull(Runs):
                if fours==0:
                    return 0
                elif fours==4:
                    return 4
                else:
                    return 6
            else:
                return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
                #sns.heatmap(dataset.isnull())
                #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

                #sns.heatmap(dataset.isnull())
                #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
                #print(dismissal)
        result=pd.get_dummies(dataset.Result)
                #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        #print(dataset)

        #res=dataset
        res = pd.DataFrame(dataset)
        message.configure(text=res)

def r_b():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        #print(dataset)

        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        #print(X)
        #print(Y)

        
        #BF vs Runs
        fig=plt.figure(figsize=(263,15))
        plt.plot('Runs',color='red')
        plt.plot('BF',color='green')
        plt.legend(['Runs','BF'],loc='best',fontsize=20)
        plt.title('BF vs Runs')
        plt.plot(dataset['Runs'])
        plt.plot(dataset['BF'])
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('Runs',fontsize=30)
        plt.show()

def fours():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)


        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values


        #fours
        fig=plt.figure(figsize=(263,15))
        plt.plot('Runs',color='red')
        plt.plot('BF',color='green')
        plt.legend(['Runs','BF'],loc='best',fontsize=20)
        plt.title('Fours')
        plt.plot(dataset['fours'])
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('Fours',fontsize=30)
        plt.show()

def b_f():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)


        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values


        #BF vs fours
        fig=plt.figure(figsize=(263,15))
        plt.plot('fours',color='red')
        plt.plot('BF',color='green')
        plt.legend(['fours','BF'],loc='best',fontsize=20)
        plt.title('number of fours')
        plt.plot(dataset['fours'],color='red')
        plt.plot(dataset['BF'],color='green')
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('fours',fontsize=30)
        plt.show()

def six():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
       

        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values



        #six
        fig=plt.figure(figsize=(263,15))
        plt.plot('Runs',color='red')
        plt.plot('BF',color='green')
        plt.legend(['Runs','BF'],loc='best',fontsize=20)
        plt.title('six')
        plt.plot(dataset['six'])
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('Six',fontsize=30)
        plt.show()


def b_s():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)


        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values


        #BF vs six
        fig=plt.figure(figsize=(263,15))
        plt.plot('six',color='red')
        plt.plot('BF',color='green')
        plt.legend(['six','BF'],loc='best',fontsize=20)
        plt.title('number of six')
        plt.plot(dataset['six'],color='red')
        plt.plot(dataset['BF'],color='green')
        plt.xlabel('BF', fontsize=30)
        plt.ylabel('six',fontsize=30)
        plt.show()


def rr():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        
        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        


        #Runs
        range=[10,20,30,40,50,60,70,80,90,100]
        fig=plt.hist(dataset.Runs,range,histtype='bar',color='darkslategrey',rwidth=0.5)
        plt.xlabel("Range of Runs",size=30)
        plt.ylabel("No. of times",size=40)
        plt.title("Runs")
        plt.show()


def f_s():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        
        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        plt.figure(figsize=(20,10))
        plt.barh(dataset.fours,dataset.six)
        plt.title("fours & six",size=30)
        plt.show()


        

def rfs():
        dataset = pd.read_csv('data.csv')
        #print(type(dataset))
        #dataset.info()
        #sns.heatmap(dataset.isnull())
        #plt.show()
        def impute_sr(cols):
                SR=cols[0]
                BF=cols[1]

                if pd.isnull(SR):

                        if BF==0:
                           return 0
                        elif BF>=25:
                            return 50
                        else:
                            return 30
                else:
                    return SR
        dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()

        #print(dataset['SR'])
        def impute_mins(cols):
                 Mins=cols[0]
                 BF=cols[1]

                 if pd.isnull(Mins):

                     if BF==0:
                         return 0
                     elif BF==50:
                         return 50
                     else:
                         return 100
                 else:
                     return Mins
        dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
                Runs=cols[0]
                fours=cols[1]

                if pd.isnull(Runs):

                     if fours==0:
                         return 0
                     elif fours==4:
                         return 4
                     else:
                         return 6
                else:
                   return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
        #sns.heatmap(dataset.isnull())
        #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

        #sns.heatmap(dataset.isnull())
        #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
        #print(dismissal)
        result=pd.get_dummies(dataset.Result)
        #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        
        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        
        fig=plt.figure(figsize=(263,15))
        plt.title('PIE')
        FOURS=sum(dataset.fours)
        SIX=sum(dataset.six)
        RUNS=(sum(dataset.Runs)-(FOURS+SIX))
        values=[FOURS,SIX,RUNS]
        label=["Fours","Six","Runs"]
        plt.pie(values,labels=label)
        plt.show()

        
def graph():
        window =Tk()
        window.title("INTERNSHIP")
        window.configure(background='white')
        window.geometry('1500x1000')
        message = tk.Label(window, text="GRAPHS AND PIE CHARTS" ,bg="white"  ,fg="dodger blue"  ,width=40  ,height=1,font=('Times New Roman', 45, 'bold underline')) 
        message.place(x=0, y=20)
        message = tk.Label(window, text="GRAPHS" ,bg="white"  ,fg="dodger blue"  ,width=40  ,height=1,font=('Times New Roman', 35)) 
        message.place(x=-100, y=140)
        message = tk.Label(window, text="PIE CHARTS" ,bg="white"  ,fg="dodger blue"  ,width=40  ,height=1,font=('Times New Roman', 35)) 
        message.place(x=470, y=190)
        def quit_window():
                window.destroy()
        x_cord = 75;
        y_cord = 20;

        
                


        trainImg = tk.Button(window, text="Range of Runs", command=rr  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=850, y=245)

        trainImg = tk.Button(window, text="Four and Six", command=f_s  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=850, y=290)
    
        trainImg = tk.Button(window, text="Runs-Fours-Six", command=rfs  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=850, y=335)
        
        

        trainImg = tk.Button(window, text="Runs vs Ball Faced", command=r_b  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=250, y=200)

        trainImg = tk.Button(window, text="Fours", command=fours  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=250, y=245)
    
        trainImg = tk.Button(window, text="Ball Faced vs Fours", command=b_f  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=250, y=290)
    
        trainImg = tk.Button(window, text="Six", command=six  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=250, y=335)
    
        trainImg = tk.Button(window, text="Ball Faced vs six", command=six  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
        trainImg.place(x=250, y=380)



        quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="dodger blue"  ,width=10  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=800, y=635-y_cord)
        window.mainloop()


def pred():
        window =tk.Tk()
        window.title("INTERNSHIP")
        window.configure(background='white')
        window.geometry('1500x1000')


        x_cord = 75;
        y_cord = 20;
        message = tk.Label(window, text="" ,bg="white"  ,fg="dodger blue"  ,width=60  ,height=15, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
        message.place(x=250-x_cord, y=70-y_cord)

       
        dataset = pd.read_csv('data.csv')
                #print(type(dataset))
                #dataset.info()
                #sns.heatmap(dataset.isnull())
                #plt.show()
        def impute_sr(cols):
            SR=cols[0]
            BF=cols[1]
            if pd.isnull(SR):
                if BF==0:
                    return 0
                elif BF>=25:
                    return 50
                else:
                    return 30
            else:
                    return SR
            dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
                #sns.heatmap(dataset.isnull())
                #plt.show()

                #print(dataset['SR'])
        def impute_mins(cols):
            Mins=cols[0]
            BF=cols[1]

            if pd.isnull(Mins):
                if BF==0:
                    return 0
                elif BF==50:
                    return 50
                else:
                    return 100
            else:
                return Mins
            dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
            Runs=cols[0]
            fours=cols[1]
            if pd.isnull(Runs):
                if fours==0:
                    return 0
                elif fours==4:
                    return 4
                else:
                    return 6
            else:
                return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
                #sns.heatmap(dataset.isnull())
                #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

                #sns.heatmap(dataset.isnull())
                #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
                #print(dismissal)
        result=pd.get_dummies(dataset.Result)
                #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        #print(dataset)
        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        #res=dataset
        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)                                              
        from sklearn.linear_model import LinearRegression
        logmodel=LinearRegression()
        logmodel.fit(X_train,Y_train)
        pred=logmodel.predict(X_test)
        #print(X_test)
        #print(pred)
        res=pred
        '''
        print(pd.DataFrame(Y_test,pred))

        from sklearn.metrics import mean_squared_error
        print(mean_squared_error(Y_test,pred))
        print(logmodel.score(X_test,Y_test))
        '''
        #res = pd.DataFrame(dataset)
        message.configure(text=res)

def acc():
        window =tk.Tk()
        window.title("INTERNSHIP")
        window.configure(background='white')
        window.geometry('1500x1000')


        x_cord = 75;
        y_cord = 20;
        message = tk.Label(window, text="" ,bg="black"  ,fg="dodger blue"  ,width=20  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
        message.place(x=550-x_cord, y=270-y_cord)

        message1 = tk.Label(window, text="" ,bg="black"  ,fg="dodger blue"  ,width=20  ,height=1, activebackground = "white" ,font=('Times New Roman', 25, ' bold ')) 
        message1.place(x=550-x_cord, y=340-y_cord)

        message2 = tk.Label(window, text="Accuracy and Error" ,bg="white"  ,fg="dodger blue"  ,width=40  ,height=1,font=('Times New Roman', 45, 'bold underline')) 
        message2.place(x=0, y=20)
        message3 = tk.Label(window, text="Error" ,bg="white"  ,fg="dodger blue"  ,width=10  ,height=1,font=('Times New Roman', 35)) 
        message3.place(x=150, y=240)
        message4 = tk.Label(window, text="Accuracy" ,bg="white"  ,fg="dodger blue"  ,width=10  ,height=1,font=('Times New Roman', 35)) 
        message4.place(x=150, y=310)


       
        dataset = pd.read_csv('data.csv')
                #print(type(dataset))
                #dataset.info()
                #sns.heatmap(dataset.isnull())
                #plt.show()
        def impute_sr(cols):
            SR=cols[0]
            BF=cols[1]
            if pd.isnull(SR):
                if BF==0:
                    return 0
                elif BF>=25:
                    return 50
                else:
                    return 30
            else:
                    return SR
            dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
                #sns.heatmap(dataset.isnull())
                #plt.show()

                #print(dataset['SR'])
        def impute_mins(cols):
            Mins=cols[0]
            BF=cols[1]

            if pd.isnull(Mins):
                if BF==0:
                    return 0
                elif BF==50:
                    return 50
                else:
                    return 100
            else:
                return Mins
            dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

        def impute_runs(cols):
            Runs=cols[0]
            fours=cols[1]
            if pd.isnull(Runs):
                if fours==0:
                    return 0
                elif fours==4:
                    return 4
                else:
                    return 6
            else:
                return Runs
        dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
                #sns.heatmap(dataset.isnull())
                #plt.show()


        dataset.drop('Ground',axis=1,inplace=True)

        dataset.dropna(inplace=True)

                #sns.heatmap(dataset.isnull())
                #plt.show()


        dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
                #print(dismissal)
        result=pd.get_dummies(dataset.Result)
                #print(result)

        dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

        dataset=pd.concat([dataset,dismissal,result],axis=1)
        #print(dataset)
        X=dataset.iloc[:,1:].values
        Y=dataset.iloc[:,0].values
        #res=dataset
        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)                                              
        from sklearn.linear_model import LinearRegression
        logmodel=LinearRegression()
        logmodel.fit(X_train,Y_train)
        pred=logmodel.predict(X_test)
        #print(X_test)
        #print(pred)

        
        #print(pd.DataFrame(Y_test,pred))

        from sklearn.metrics import mean_squared_error
       # print(mean_squared_error(Y_test,pred))
        #print(logmodel.score(X_test,Y_test))

        
        
        res = mean_squared_error(Y_test,pred)
        res1 = logmodel.score(X_test,Y_test)
        message.configure(text=res)
        message1.configure(text=res1)

        def quit_window():
                window.destroy()

        quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="dodger blue"  ,width=10  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=800, y=635-y_cord)

def menu():
    window =tk.Tk()
    window.title("INTERNSHIP")
    window.configure(background='white')
    window.geometry('1500x1000')
    

    x_cord = 75;
    y_cord = 20;

    
    def quit_window():
           window.destroy()
 
    
    message = tk.Label(window, text="CIRCKETER DATASET" ,bg="white"  ,fg="dodger blue"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
    message.place(x=170, y=20)

    trainImg = tk.Button(window, text="CIRCKETER DATASET", command=dataset  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=150)

    trainImg = tk.Button(window, text="ALTER DATASET", command=alter  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=250)
    
    trainImg = tk.Button(window, text="GRAPHS & PIE CHARTS", command=graph  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=350)
    
    trainImg = tk.Button(window, text="PREDICTED VALUES", command=pred  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=450)
    
    trainImg = tk.Button(window, text="ACCURACY", command=acc  ,fg="black"  ,bg="dodger blue"  ,width=25  ,height=1, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    trainImg.place(x=550, y=550)


    quitWindow = tk.Button(window, text="BACK", command=quit_window  ,fg="black"  ,bg="dodger blue"  ,width=10  ,height=2, activebackground = "blue" ,font=('Times New Roman', 15, ' bold '))
    quitWindow.place(x=1000, y=735-y_cord)

    window.mainloop()


        
class start:
    def __init__(self):
        window =Tk()
        window=Canvas(window,width=1500,height=1000,background='white')
        window.pack()
        image = PhotoImage(file = '12.png')
        window.create_image(400,150, anchor = NW, image = image)
        message = tk.Label(window, text="CIRCKETER PREFORMANCE PREDICTION" ,bg="white"  ,fg="dodger blue"  ,width=40  ,height=1,font=('Times New Roman', 45, 'bold underline')) 
        message.place(x=0, y=20)
        message = tk.Label(window, text="USING MACHINE LEARNING ALGORITHM" ,bg="white"  ,fg="dodger blue"  ,width=60  ,height=1,font=('Times New Roman', 25, 'bold underline')) 
        message.place(x=100, y=100)
       # imagetest = PhotoImage(file='C:\\Users\\Home-PC\\Desktop\\final\\images\\2.png')
       # window.create_image(250, 400, image=imagetest)
       # imagetest1 = PhotoImage(file='C:\\Users\\Home-PC\\Desktop\\final\\images\\1.png')
        #window.create_image(1200, 400, image=imagetest1)
        x_cord = 75;
        y_cord = 20;

        def quit_window():
            MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
            if MsgBox == 'yes':
                tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
                window.destroy()




        quitWindow = tk.Button(window, text="Start", command=menu  ,fg="black"  ,bg="dodger blue",width=15  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=500, y=635-y_cord)        

        quitWindow = tk.Button(window, text="QUIT", command=quit_window  ,fg="black"  ,bg="dodger blue"  ,width=10  ,height=2, activebackground = "BLUE" ,font=('Times New Roman', 15, ' bold '))
        quitWindow.place(x=800, y=635-y_cord)
        window.mainloop()



if(1):
     GUUEST=start()







