from csv import DictReader, DictWriter
import csv
import urllib
import pandas_datareader.data as web
import os.path

class StockDataAcquire(object):
    def __init__(self):
        self.DefultStockList='nasdaqlisted.txt'

    def GetStockSymbleList(self, stock_listfilename=None):
        '''
        use the default file list if not specified

        return list of symbles
        '''
        stockList=[]
        if stock_listfilename is None:
            stock_listfilename=self.DefultStockList
        
        '''
        Load the file and decode the data
        '''
        with open(stock_listfilename) as Stock_list_io:
            Stock_list_filename_lines=Stock_list_io.readlines()

        # decoder the header, and get the data strcuture of the csv file
        Stock_header_infor=Stock_list_filename_lines[0].split('|')
        Stock_name_Symbol_position=Stock_header_infor.index('Symbol')
        Stock_name_Security_name_position=Stock_header_infor.index('Security Name')
        Stock_name_Market_Category_position=Stock_header_infor.index('Market Category')
        Stock_name_Financial_Status_position=Stock_header_infor.index('Financial Status')
        Stock_name_ETF_position=Stock_header_infor.index('ETF')

        print(Stock_header_infor)
        '''
        finish decode the csv compony name list, start aquire the realtime data
        loops on all the lines and get all the price, create a folder and save the price in that folder
        '''

        self.iterator_counter=0
        
        for Stock_list_Single_line in Stock_list_filename_lines[:-1]:
            stockInfor={}
            Stock_Single_line_infor=Stock_list_Single_line.split('|')
            print(self.iterator_counter)
            self.iterator_counter=self.iterator_counter+1
            print (Stock_Single_line_infor[Stock_name_Symbol_position]+'  Stock_Market_Category  '+Stock_Single_line_infor[Stock_name_Market_Category_position])            
            stockInfor['Symbol']=Stock_Single_line_infor[Stock_name_Symbol_position]
            stockInfor['SecurityName']=Stock_Single_line_infor[Stock_name_Security_name_position]
            stockInfor['SecurityCategory']=Stock_Single_line_infor[Stock_name_Market_Category_position]
            stockInfor['FinalcialStatus']=Stock_Single_line_infor[Stock_name_Financial_Status_position]
            stockInfor['ETF']=Stock_Single_line_infor[Stock_name_ETF_position]
            stockList.append(stockInfor)

        return stockList
    
    
    def Save2csv(self,Dic=[],saveName=None):
        pass

    def GetHistoricData(self,stockList=[]):
        '''
        Check whether the file have been downloaded, if not download the data file
        '''
        counter=0
        for stockInfor in self.GetStockSymbleList()[1:]:
            saveFname='Data/'+stockInfor['Symbol']+".csv"
            if not os.path.isfile(saveFname):
                print(counter)
                print(saveFname)
                print('     ... Getting data')
                try:
                    df = web.DataReader(stockInfor['Symbol'],'quandl','1980-01-01', '2019-11-05')
                    df.to_csv(saveFname)
                    print('     ... success ')
                except :
                    print('     ... error ')
                    pass
            counter=counter+1
    
    def GetHistoricData1(self,stockList=[]):
        '''
        Check whether the file have been downloaded, if not download the data file
        '''
        counter=0
        for stockInfor in stockList:
            stockSymbol=stockInfor
            saveFname='Data/'+stockSymbol+".csv"
            if not os.path.isfile(saveFname):
                print(counter)
                print(saveFname)
                print('     ... Getting data')
                try:
                    df = web.DataReader(stockSymbol,'quandl','1980-01-01', '2019-11-05')
                    df.to_csv(saveFname)
                    print('     ... success ')
                except :
                    print('     ... error ')
                    pass
            counter=counter+1
    
    def GetNasdaq100IndexNDX(self,fname=None):
        if fname is None:
            fname="nasdaq100.csv"
        with open(fname,'rt') as f:
            reader=csv.reader(f)
            listf=list(reader)
        b=[el[1] for el in listf]
        return b
    def IsExist(self, filename=None):
        if filename is None:
            return True
        else:
            pass
        

    def Log(self):
        pass

if __name__ == "__main__":
    test = StockDataAcquire()
    a=test.GetStockSymbleList()
    print(a)
    # test.GetHistoricData()
    test.GetNasdaq100IndexNDX()
    test.GetHistoricData1(stockList=test.GetNasdaq100IndexNDX())
