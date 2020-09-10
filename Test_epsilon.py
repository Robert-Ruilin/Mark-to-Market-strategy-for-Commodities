# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 08:38:52 2020

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import FuncFormatter  

class Epsilon_calibration():
    
    identity = ['ag2101', 'ag2102', 'ag2103', 'ag2104', 'ag2105', 'au2102', 'au2104', 'au2108']  #交易涉及合约
    start = ['210000', '090000', '103000', '133000']  #开市时间
    end = ['023000', '101500', '113000', '150000']  #闭市时间
        
    def __init__(self, dir_name):
        self.file_name = os.listdir(dir_name)  #所有行情数据原始文件名称列表（时段：天）
        self.file_dir = [os.path.join(dir_name, x) for x in self.file_name]  #所有行情数据原始文件路径列表（时段：天）
    
    def split_set(self, data):
        data['行情时间数字格式'] = data['行情时间'].apply(lambda x: int(x[0:2]+x[3:5]+x[6:8]))
        data_set_identity = []
        for i in range(len(self.identity)):  #将每日行情数据按照合约种类拆分为8个子集 （4）
            data_split = data[data['合约'] == self.identity[i]]
            data_set_identity.append(data_split)
            
        data_set_time = [[] for i in range(len(self.identity))]
        for i in range(len(self.identity)):  #将每日行情数据按照时段区间拆分为4个子集 （8*4）
            for j in range(len(self.start)):
                if int(self.start[j]) < int(self.end[j]):
                    data_interval = data_set_identity[i].loc[(data_set_identity[i]['行情时间数字格式']>int(self.start[j])) & (data_set_identity[i]['行情时间数字格式']<int(self.end[j])),:]
                else:
                    data_interval = data_set_identity[i].loc[(data_set_identity[i]['行情时间数字格式']>int(self.start[j])) | (data_set_identity[i]['行情时间数字格式']<int(self.end[j])),:]
                data_set_time[i].append(data_interval)
            
        return data_set_time  # 8 identities * 4 time intervals      

    def get_epsilon(self, data_set_time, N = 30):  #计算日内每种合约4个开闭市时段的epsilon列表
        Epsilon_1day = []
        for i in range(len(self.identity)):
            for j in range(len(self.start)):
                data_set_time[i][j]['收益率'] = (data_set_time[i][j]['最新价']-data_set_time[i][j]['最新价'].shift(1))/data_set_time[i][j]['最新价'].shift(1)
                data_set_time[i][j]['30tick收益均值'] = np.round(data_set_time[i][j]['收益率'].rolling(window=N).mean(),10)
                data_set_time[i][j]['30tick收益标准差'] = np.round(data_set_time[i][j]['收益率'].rolling(window=N).std(),10)
                data_set_time[i][j]['epsilon'] = abs(data_set_time[i][j]['收益率'])/data_set_time[i][j]['30tick收益标准差']
                #data_set_time[i][j]['epsilon'].fillna('该时点无法测算', inplace = True)
                data_set_time[i][j] = data_set_time[i][j][['合约','行情时间','行情时间数字格式','最新价',
                                                           '收益率','30tick收益均值','30tick收益标准差','epsilon']]
            epsilon_concat = pd.concat(data_set_time[i], axis = 0)
            Epsilon_1day.append(epsilon_concat)
        
        return Epsilon_1day  # 8 identities
    
    def all_epsilon(self):  #得到指定时段内所有的epsilon列表
        Epsilon = []
        T = len(self.file_dir)
        for i in range(T):
            raw_data = pd.read_csv(self.file_dir[i], encoding = 'GB2312')
            data_set = self.split_set(raw_data)
            Epsilon_1day = self.get_epsilon(data_set, N = 30)
            Epsilon.append(Epsilon_1day)
            
        return Epsilon  # T days * 8 identities

    def epsilon_analysis(self, data, filter_0 = False):  #得到epsilon的分布和大小
        Tbepsilon_set = []
        if filter_0:  #filter_0开关用来过滤 epsilon = 0的情况
            for i in range(len(data)):
                data[i] = data[i][data[i]['epsilon'] != 0]    
        for i in range(len(data)):
            table_epsilon = pd.DataFrame(index = range(1), columns = ['日期','均值','epsilon=0占比','0<epsilon<=1占比','1<epsilon<=2占比','2<epsilon<=3占比','3<epsilon<=4占比',
                                                                                    '4<epsilon<=5占比','epsilon>5占比','1/4分位点','中位数','3/4分位点','vol of epsilon','合约'])
            data[i] = data[i].dropna(subset = ['epsilon'])
            table_epsilon.iloc[0,1] = data[i]['epsilon'].mean()  #计算epsilon均值
            table_epsilon.iloc[0,2] = '%.2f%%' % (data[i]['epsilon'][data[i]['epsilon']==0].count()/data[i]['epsilon'].count() * 100)  #计算epsilon = 0占比
            table_epsilon.iloc[0,3] = '%.2f%%' % (data[i]['epsilon'][(data[i]['epsilon']>0) & (data[i]['epsilon']<=1)].count()/data[i]['epsilon'].count() * 100)  #计算epsilon在(0,1]区间占比
            table_epsilon.iloc[0,4] = '%.2f%%' % (data[i]['epsilon'][(data[i]['epsilon']>1) & (data[i]['epsilon']<=2)].count()/data[i]['epsilon'].count() * 100) #计算epsilon在(1,2]区间占比
            table_epsilon.iloc[0,5] = '%.2f%%' % (data[i]['epsilon'][(data[i]['epsilon']>2) & (data[i]['epsilon']<=3)].count()/data[i]['epsilon'].count() * 100) #计算epsilon在(2,3]区间占比
            table_epsilon.iloc[0,6] = '%.2f%%' % (data[i]['epsilon'][(data[i]['epsilon']>3) & (data[i]['epsilon']<=4)].count()/data[i]['epsilon'].count() * 100) #计算epsilon在(3,4]区间占比
            table_epsilon.iloc[0,7] = '%.2f%%' % (data[i]['epsilon'][(data[i]['epsilon']>4) & (data[i]['epsilon']<=5)].count()/data[i]['epsilon'].count() * 100) #计算epsilon在(4,5]区间占比
            table_epsilon.iloc[0,8] = '%.2f%%' % (data[i]['epsilon'][data[i]['epsilon']>5].count()/data[i]['epsilon'].count() * 100) #计算epsilon在(5,+infinity)区间占比
            table_epsilon.iloc[0,9] = data[i]['epsilon'].quantile(0.25)  #计算epsilon25%分位点
            table_epsilon.iloc[0,10] = data[i]['epsilon'].quantile(0.5)  #计算epsilon中位数
            table_epsilon.iloc[0,11] = data[i]['epsilon'].quantile(0.75)  #计算epsilon75%分位点
            table_epsilon.iloc[0,12] = data[i]['epsilon'].std()  #计算epsilon的波动率
            table_epsilon.iloc[0,13] = self.identity[i]  #标注合同种类
            Tbepsilon_set.append(table_epsilon)
            
        return Tbepsilon_set  # 8 identities * 1 column
    
    def output_epsilon_table(self):  #将epsilon的历史数据导出至excel
        Epsilon = self.all_epsilon()
        table_set = []
        for i in range(len(self.file_name)):
            Tbepsilon = self.epsilon_analysis(Epsilon[i], filter_0 = False)
            table_set.append(Tbepsilon)
        date = self.file_name
        for i in range(len(date)):
            date[i] = date[i][3:11]
        daily_table = []
        for i in range(len(date)):
            daily_set = pd.concat(table_set[i], axis = 0)
            daily_set['日期'] = date[i]
            daily_table.append(daily_set)
        Summary = pd.concat(daily_table, axis = 0)
        Summary = Summary.dropna(subset = ['均值'])
        output_list = []
        for i in range(len(self.identity)):
            df_identity = Summary[Summary['合约'] == self.identity[i]]
            output_list.append(df_identity)
        
        output = pd.ExcelWriter('E:\HRL\epsilon_table.xlsx')  #保存路径
        for i in range(len(self.identity)):
            output_list[i].to_excel(output, sheet_name = self.identity[i], index = False)
        output.save()
        
        return output_list  # 8 identities * T days
    
    def price_chart(self, output_list):  #绘制epsilon在指定时段内的波动情况
        for i in range(len(output_list)):
            fig, left_axis = plt.subplots(figsize = (20, 12))
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            X = output_list[i]['日期']        
            Y1 = output_list[i]['均值']
            p1, = left_axis.plot(X, Y1, 'ro-', label = 'Epsilon每日均值')
            
            right_axis = left_axis.twinx()
            Y2 = output_list[i]['0<epsilon<=1占比'].apply(lambda x: float(x.strip('%'))/100)
            Y3 = output_list[i]['1<epsilon<=2占比'].apply(lambda x: float(x.strip('%'))/100)
            Y_sum = output_list[i]['epsilon=0占比'].apply(lambda x: 1-float('0.'+x[0:2]+x[3:5]))
            Y4 = Y_sum - Y2 - Y3
            right_axis.bar(X, np.array(Y2), width = 0.8, facecolor='#9999ff', alpha = 0.6, label = 'Epsilon每日落在(0,1]的比例')
            right_axis.bar(X, np.array(Y3), width = 0.8, bottom = np.array(Y2), facecolor='b', alpha = 0.6, label = 'Epsilon每日落在(1,2]的比例')
            right_axis.bar(X, np.array(Y4), width = 0.8, bottom = np.array(Y3+Y2), facecolor='#00BFFF', alpha = 0.6, label = 'Epsilon每日落在(2,+infty)的比例')   
            
            left_axis.set_xticklabels(X, rotation = 30)
            #x_major_locator = MultipleLocator(2)
            #left_axis.xaxis.set_major_locator(x_major_locator)
            
            right_axis.set_ylim(0,math.ceil(max(Y_sum)*20)/20+0.01)  
            right_axis.set_yticks(np.arange(0,math.ceil(max(Y_sum)*20)/20+0.01,math.ceil(max(Y_sum)*20)/200))
            def to_percent(temp, position):
                return '%.1f%%' %(100*temp)
            right_axis.yaxis.set_major_formatter(FuncFormatter(to_percent))
            left_axis.set_title('Epsilon波动图'+'——'+self.identity[i]+'合约', fontsize = 20)
            left_axis.set_xlabel('日期',fontsize = 14)
            left_axis.set_ylabel('Epsilon每日均值',fontsize = 14)
            
            left_axis.legend(loc=(.02,.92), fontsize = 12)
            right_axis.legend(loc=(.02,.80), fontsize = 12) 
            
            plt.savefig(self.identity[i]+'.png')
        plt.show()
        
        return
                    

'''-------------------------------------------------------------------------------------------------------------------------------------------'''

if __name__ == '__main__':    
    dir_name = r'E:\HRL\Analysis\Analysis'  #文件读取路径
    Test = Epsilon_calibration(dir_name)
    table = Test.output_epsilon_table()
    Test.price_chart(table)


