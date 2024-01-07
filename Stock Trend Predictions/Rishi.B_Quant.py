# RISHI BHARADWAJ
#2022A7PS2001H


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_df(file):
    # input: file path to a csv file
    # returns a df and axs to a plot
    df = pd.read_csv(file)
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_15"] = df["Close"].rolling(15).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    fig1,axs=plt.subplots(2,3,figsize=(20,10))
    
    df.plot(title="SMA",x="Date",y="SMA_15",color="red",ax=axs[0][0],ylabel="SMA_15")
    df.plot(x="Date",y="SMA_50",color="blue",ax=axs[0][0])
    df.plot(title="EMA",x="Date",y="EMA_5",color="blue",ax=axs[0][1],ylabel="EMA")
    df.plot(x="Date",y="EMA_10",color="red",ax=axs[0][1],ylabel="EMA")
    
    obv=np.where(df["Close"]>df["Close"].shift(1),df["Volume"],-(df["Volume"])).cumsum()
    df["OBV"]=obv
    df.plot(title="OBV",x="Date",y="OBV",ax=axs[1][0],ylabel="OBV")

    stochastic(df,14)
    df.plot(title="Stochastic",x="Date",y="Fast_k",color="red",ax=axs[1][1],ylabel="%K and %D")
    df.plot(x="Date",y="D",color="green",ax=axs[1][1])
    
    df["20"]=20
    df["80"]=80
    # df["SMA_I"]=np.nan
 
    # for i in range(len(df)):
    #     if (np.abs(df["SMA_15"][i]-df["SMA_50"][i])<1.05):
    #         df["SMA_I"][i]=df["SMA_15"][i]

    # df.plot(x="Date",y="SMA_I",ax=axs[0][0],color="black",kind="scatter",ylabel="SMA")
    df.plot(x="Date",y="20",ax=axs[1][1],color="black")
    df.plot(x="Date",y="80",ax=axs[1][1],color="blue")
    #plt.show()
    return df,axs
    
    
    
    

def stochastic(df,period):
    # input parameters: df and time period
    # modifies the df to add a column "Stochastic"
    df["Fast_k"]=np.nan
    for i in range(len(df)):
        low=df["Close"][i]
        high=df["Close"][i]+1
        if i>=period:
            low = df["Close"][i]
            high = df["Close"][i]
            s = 0
            while(s<period):
                if df["Close"][i-s]< low:
                    low=df["Close"][i-s]
                elif df["Close"][i-s]> high:
                    high=df["Close"][i-s]
                s+=1
            c = df["Close"][i]
            k= (c-low)*100/(high-low)
            df["Fast_k"][i]=k
            df["D"] = df["Fast_k"].rolling(3).mean()







#Multiple indicators:
#Buy when the  Stochatic  crosses above 20 and OBV is moving up and the close is above 50 MA / 10 MA
#Sell when the  Stochatic  crosses below 80 and OBV is moving down and the close is below 50 MA / 10MA
def make_trades(df,axs):
    # input parameters: dataframe, axes to a plot
    # returns: profit percentage
    df["Buy"]=np.nan
    df["Sell"]=np.nan
    df["Bought_at"]=np.nan
    df["Sold_at"]=np.nan
    df["Realised Gains"]=0
    amount=1000
    df["Value"]=1000
    
    #Buy when the  Stochastic  crosses above 20 and OBV is moving up and the close is above 50 SMA and 15 SMA
    def buy(df,i,amount,shares):
        if df["Fast_k"][i] >df["20"][i] and (df["Fast_k"][i-1] or df["Fast_k"][i-2] or df["Fast_k"][i-3])<df["20"][i]:
            if df["OBV"][i]>(df["OBV"][i-1] or df["OBV"][i-1] or df["OBV"][i-1]):
                if df["Close"][i]>df["SMA_50"][i] and df["Close"][i]>df["SMA_15"][i]:
                        df["Buy"][i]=df["Fast_k"][i]
                        df["Bought_at"][i]=df["Close"][i]
                        shares=amount/df["Bought_at"][i]
                        return 1,shares,df,0
                else:
                   return 0,shares,df,amount
            else:
                return 0,shares,df,amount
        else:
            return 0,shares,df,amount
                
    #Sell when the  Stochastic  crosses below 80 and OBV is moving down and the close is below 50 SMA and 15 SMA
    def sell(df,i,shares,amount):
        if df["Fast_k"][i]<df["80"][i] and (df["Fast_k"][i-1] or df["Fast_k"][i-2] or df["Fast_k"][i-3])>df["80"][i]:
            if df["OBV"][i]<(df["OBV"][i-1] or df["OBV"][i-1] or df["OBV"][i-1]):
                if (df["Close"][i]<df["SMA_50"][i]) and (df["Close"][i]<df["SMA_15"][i]):
                    df["Sell"][i]=df["Fast_k"][i]
                    df["Sold_at"][i]=df["Close"][i]
                    amount=df["Close"][i]*shares
                    return 0,0,df,amount
                else:
                     return 1,shares,df,amount
            else:
               return 1,shares,df,amount
        else:
            return 1,shares,df,amount
    def trade(df,amount):
        i=1
        b=0
        realised_gains=0
        shares=0
        amount_i=amount
        while(i<len(df) and df["Value"][i]>0):
            while(b==0 and i<len(df) ):
                b,shares,df,amount=buy(df,i,amount,shares)
                df["Value"][i]=shares*df["Close"][i] +amount
                df["Realised Gains"][i]=realised_gains
                i+=1
            while(b==1 and i<len(df)):
                b,shares,df,amount=sell(df,i,shares,amount)
                df["Value"][i]=shares*df["Close"][i] + amount
                df["Realised Gains"][i]=realised_gains
                if shares==0:
                    df["Realised Gains"][i]=df["Value"][i] - amount_i
                    realised_gains=df["Realised Gains"][i]
                i+=1
        return df
    
    df=trade(df,amount)
    
    
    #df.plot(x="Date",y="Bought_at",ax=axs[0][0],color="blue",kind="scatter") #Plots buy and sell points on the SMA graph
    #df.plot(x="Date",y="Sold_at",ax=axs[0][0],color="purple",kind="scatter") #Plots buy and sell points on the SMA graph
    df.plot(title= "Value",x="Date",y="Value",ax=axs[0][2],color="purple",ylabel="Value")
    df.plot(title="Realised Gains",x="Date",y="Realised Gains",ax=axs[1][2],color="green",ylabel="Realised Gains")
    
    plt.show()
    profit=(df["Value"][len(df)-1]-amount)/amount
    profit_percent=profit*100

    return profit_percent


# For a csv file of csv file paths: 
# def main():    
    
    #returns a dataframe of stocks and profit percentages
    
    # list=pd.read_csv(r"C:\Users\rishi\Studies\CS\WSC\list.csv")
    # f_df=pd.DataFrame(index=range(13))
    # f_df["Stock"]=np.nan
    # f_df["Profit"]=np.nan
    # f_df["Negative"]=0
    # f_df["Positive"]=0
    # overall=0
    # for i in range(len(list)):
    #     path=list["File Path"][i]
    #     df,axs= make_df(path)
    #     profit_percent=make_trades(df,axs)
    #     overall+=profit_percent
    #     overall/=len(list)
    #     f_df["Stock"][i]=list["Stock"][i]
    #     f_df["Profit"][i]=profit_percent
    # for i in range (len(f_df)):
    #     if f_df["Profit"][i]>0:
    #         f_df["Positive"][i]=f_df["Profit"][i]
    #         f_df["Negative"][i]=0
    #     else:
    #         f_df["Negative"][i]=f_df["Profit"][i]
    #         f_df["Positive"][i]=0
            
    # fig2=plt.figure(figsize=(20,12))
    # f_df.plot(x="Stock",y="Positive",kind="bar",ax=fig2.gca(),color="green",fontsize="15")
    # f_df.plot(x="Stock",y="Negative",kind="bar",ax=fig2.gca(),color="red",fontsize="15")
    # plt.ylabel("Results", fontsize=16)
    # plt.xlabel("Stocks", fontsize=16)
    # plt.title("Results",fontsize=18)
    # plt.tight_layout()
    # plt.show()
    # print("Overall:"+str(overall))
    # print(f_df.sort_values(by=["Profit"],ascending=False).head(13))


#For a single csv:  
def main():
    # displays the plots and profit percentage 
    df,axs=make_df(r"C:\Users\rishi\Studies\CS\WSC\TATASTEEL.NS.csv")
    profit_percent=make_trades(df,axs)
    print("Profit:" + str(profit_percent) +"%")
    
main()
