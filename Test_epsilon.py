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
from matplotlib.pyplot import MultipleLocator
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

    def output_price(self):  #合并并且导出所有价格数据
        total = []
        T = len(self.file_dir)
        for i in range(T):
            raw_data = pd.read_csv(self.file_dir[i], encoding = 'GB2312')
            raw_data['日期'] = self.file_name[i][3:11]
            data_set = self.split_set(raw_data)
            sub_total = []
            for j in range(len(self.identity)):
                sub_set = pd.concat(data_set[j], axis = 0)
                sub_total.append(sub_set)
            sub = pd.concat(sub_total, axis = 0)
            total.append(sub)
        df_total = pd.concat(total, axis = 0)
        df_total = df_total[['日期','合约','行情时间','最新价']]
        df_total.to_csv('E:\HRL\.csv', index = False, encoding='utf_8_sig')
        return df_total      

    def get_epsilon(self, data_set_time, N = 30):  #计算日内每种合约4个开闭市时段的epsilon列表
        Epsilon_1day = []
        for i in range(len(self.identity)):
            for j in range(len(self.start)):
                data_set_time[i][j]['收益率'] = (data_set_time[i][j]['最新价']-data_set_time[i][j]['最新价'].shift(1))/data_set_time[i][j]['最新价'].shift(1)
                data_set_time[i][j]['30tick收益均值'] = np.round(data_set_time[i][j]['收益率'].shift(1).rolling(window=N-1).mean(),10)
                data_set_time[i][j]['30tick收益标准差'] = np.round(data_set_time[i][j]['收益率'].shift(1).rolling(window=N-1).std(ddof = 0),10)
                data_set_time[i][j].loc[data_set_time[i][j]['30tick收益标准差'] == 1e-10, '30tick收益标准差'] = 0  #bug? std(29个0) = 1e-10
                data_set_time[i][j]['epsilon'] = abs(data_set_time[i][j]['收益率'])/data_set_time[i][j]['30tick收益标准差']
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
    
    def daily_market_trend(self, Epsilon, date, name):  #观察每日合约价格波动情况
        date_set = self.file_name
        for i in range(len(date_set)):
            date_set[i] = date_set[i][3:11]
        for i in range(len(date)):
            id_date = date_set.index(date[i])
            id_name = self.identity.index(name)
            Y = Epsilon[id_date][id_name]['最新价'][1:]
            X = Epsilon[id_date][id_name]['行情时间'][1:]
            fig, ax = plt.subplots(2,1)
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            ax[0].plot(X, Y, '-', label = '日内价格变动')
            
            split_time = [23000, 101500, 113000]
            for j in range(3):
                num = Epsilon[id_date][id_name]['行情时间数字格式'][(Epsilon[id_date][id_name]['行情时间数字格式']>split_time[j]) & (Epsilon[id_date][id_name]['行情时间数字格式']<210000)].values[0]
                id_split = Epsilon[id_date][id_name]['行情时间数字格式'].tolist().index(num)
                x_line = Epsilon[id_date][id_name]['行情时间'].values[id_split]
                ax[0].vlines(x_line, Epsilon[id_date][id_name]['最新价'][1:].min(), Epsilon[id_date][id_name]['最新价'][1:].max(), color = 'red', linestyles = 'dashed', linewidth = 1.5)
            
            x_major_locator = MultipleLocator(2000)
            ax[0].xaxis.set_major_locator(x_major_locator)
            ax[0].set_ylabel('价格')
            ax01 = ax[0].twinx()
            Epsilon[id_date][id_name].loc[Epsilon[id_date][id_name]['epsilon'] == np.inf, 'epsilon'] = 0
            Epsilon[id_date][id_name].loc[Epsilon[id_date][id_name]['epsilon'].isnull(), 'epsilon'] = 0
            Y2 = Epsilon[id_date][id_name]['epsilon'][1:]
            ax01.plot(X, Y2, 'c-', alpha = 0.4, label = '日内epsilon变动')
            ax01.xaxis.set_major_locator(x_major_locator)
            ax01.set_ylabel('Epsilon数值')
            ax[0].legend(bbox_to_anchor=(1, 1))
            ax01.legend(bbox_to_anchor=(1, 0.9))
            
            Y1 = Epsilon[id_date][id_name]['收益率'][1:]
            ax[1].plot(X, Y1, '-', label = '日内价格收益率变动')
            x_major_locator = MultipleLocator(2000)
            ax[1].xaxis.set_major_locator(x_major_locator)
            def to_percent(temp, position):
                return '%.2f%%' %(100*temp)
            ax[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
            ax[1].set_ylim(-0.006,0.006)
            ax[1].set_xlabel('日期')
            ax[1].set_ylabel('收益率（%）')
            plt.suptitle(date[i]+' '+name+'合约价格日内波动', fontsize = 20)
        plt.show()
            
    def epsilon_filter_chart(self, Epsilon):  #观察合约epsilon类别： {不等于0，等于0，无法计算}
        total = []
        for i in range(len(self.file_dir)):
            df = pd.concat(Epsilon[i], axis = 0)
            total.append(df)
        total_data = pd.concat(total, axis = 0)
        total_data['epsilon类别'] = 'epsilon不等于0（价格变动，sigma不等于0）'
        total_data.loc[total_data['epsilon'] == 0, 'epsilon类别'] = 'epsilon等于0（价格不变，sigma不等于0）'
        total_data.loc[total_data['epsilon'] == np.inf, 'epsilon类别'] = 'epsilon无法计算（价格变动，sigma等于0）'
        total_data.loc[total_data['epsilon'].isnull(), 'epsilon类别'] = 'epsilon无法计算（价格不变，sigma等于0）'
        for i in range(len(self.identity)):
            data_set = total_data['epsilon类别'][total_data['合约']==self.identity[i]]
            count = data_set.value_counts()
            plt.figure(figsize = (20,12))
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            data = count
            labels = count.index
            colors = ['#BC8F8F','#9999ff','blue','#00BFFF']
            explode = (0,0,0.1,0)
            plt.axes(aspect = 'equal')
            plt.xlim(0,8)
            plt.ylim(0,8)
            plt.pie(x = data, labels = labels, explode = explode, colors = colors, autopct = '%.1f%%',
                    pctdistance = 0.8, labeldistance=10, startangle = 180, center = (4,4),
                    radius = 3.8, wedgeprops = {'linewidth':1, 'edgecolor':'black'},
                    textprops = {'fontsize':12,'color':'w'}, frame = 1)
            plt.xticks(())
            plt.yticks(())
            plt.title(self.identity[i]+'合约的epsilon类别', fontsize = 20)
            plt.legend(bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0, fontsize = 12)
            plt.savefig(self.identity[i]+'epsilon类别.png')
        plt.show()
            

    def epsilon_analysis(self, data, filter_0 = False):  #得到epsilon的分布和大小
        Tbepsilon_set = []
        if filter_0:  #filter_0开关用来过滤 epsilon = 0的情况
            for i in range(len(data)):
                data[i] = data[i][data[i]['epsilon'] != 0]    
        for i in range(len(data)):
            table_epsilon = pd.DataFrame(index = range(1), columns = ['日期','均值','epsilon=0占比','0<epsilon<=1占比','1<epsilon<=2占比','2<epsilon<=3占比','3<epsilon<=4占比',
                                                                      '4<epsilon<=5占比','epsilon>5占比','1/4分位点','中位数','3/4分位点','vol of epsilon','合约'])
            data[i] = data[i].dropna(subset = ['epsilon'])
            data[i] = data[i][data[i]['epsilon'] != np.inf]
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
    
    def epsilon_chart(self, output_list):  #绘制epsilon在指定时段内的波动情况
        for i in range(len(output_list)):
            fig, left_axis = plt.subplots(figsize = (20, 12))
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            X = output_list[i]['日期']        
            Y1 = output_list[i]['均值']
            Y2 = output_list[i]['0<epsilon<=1占比'].apply(lambda x: float(x.strip('%'))/100)
            Y3 = output_list[i]['1<epsilon<=2占比'].apply(lambda x: float(x.strip('%'))/100)
            Y0 = output_list[i]['epsilon=0占比'].apply(lambda x: float(x.strip('%'))/100)
            Y_sum = output_list[i]['epsilon=0占比'].apply(lambda x: 1-float(x.strip('%'))/100)
            Y4 = Y_sum - Y2 - Y3
            left_axis.bar(X, np.array(Y0), width = 0.8, facecolor='#BC8F8F', alpha = 0.6, label = 'Epsilon每日等于0的比例')
            left_axis.bar(X, np.array(Y2), width = 0.8, bottom = np.array(Y0), facecolor='#9999ff', alpha = 0.6, label = 'Epsilon每日落在(0,1]的比例')
            left_axis.bar(X, np.array(Y3), width = 0.8, bottom = np.array(Y2+Y0), facecolor='b', alpha = 0.6, label = 'Epsilon每日落在(1,2]的比例')
            left_axis.bar(X, np.array(Y4), width = 0.8, bottom = np.array(Y3+Y2+Y0), facecolor='#00BFFF', alpha = 0.6, label = 'Epsilon每日落在(2,+infty)的比例')   
            right_axis = left_axis.twinx()
            right_axis.plot(X, Y1, 'ro-', label = 'Epsilon每日均值')
            left_axis.set_xticklabels(X, rotation = 40)        
            #right_axis.set_ylim(0,math.ceil(max(Y_sum)*20)/20+0.01)  
            #right_axis.set_yticks(np.arange(0,math.ceil(max(Y_sum)*20)/20+0.01,math.ceil(max(Y_sum)*20)/200))
            def to_percent(temp, position):
                return '%.1f%%' %(100*temp)
            left_axis.yaxis.set_major_formatter(FuncFormatter(to_percent))
            left_axis.set_title('Epsilon波动图'+'——'+self.identity[i]+'合约', fontsize = 20)
            left_axis.set_xlabel('日期',fontsize = 14)
            left_axis.set_ylabel('百分比',fontsize = 14)
            left_axis.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize = 12)
            right_axis.legend(bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0, fontsize = 12)
            fig.subplots_adjust(right=0.75)        
            plt.savefig(self.identity[i]+'.png')
        plt.show()
        return
                    

'''-------------------------------------------------------------------------------------------------------------------------------------------'''


if __name__ == '__main__':    
    dir_name = r'E:\HRL\Analysis\Analysis'  #文件读取路径
    Test = Epsilon_calibration(dir_name)
    #table = Test.output_epsilon_table()
    #Epsilon = Test.all_epsilon()
    #Test.epsilon_chart(table)
    #mkt = Test.market_trend(Test.all_epsilon())
    date = ['20200824']
    name = 'ag2101'
    Test.daily_market_trend(Epsilon, date, name)
