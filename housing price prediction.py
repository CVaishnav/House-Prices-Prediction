from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
train_url="C:/Users/dell/Downloads/train.csv"
train=pd.read_csv(train_url)
#a=train
test_url="C:/Users/dell/Downloads/test.csv"
test=pd.read_csv(test_url)
target=train["SalePrice"]

train["MSZoning"].replace(["A","C (all)","FV","I","RH","RL","RP","RM",None],[40,60,50,30,70,90,100,80,0],inplace=True)
train["Street"].replace(["Pave","Grvl",None],[100,50,0],inplace=True)
train["LandContour"].replace(["Lvl","Bnk","HLS","Low",None],[100,75,50,20,0],inplace=True)
train["Utilities"].replace(["AllPub","NoSewr","NoSeWa","ELO",None],[100,60,30,10,0],inplace=True)
train["LandSlope"].replace(["Gtl","Mod","Sev",None],[100,60,20,0],inplace=True)
train["BldgType"].replace(["1Fam","2fmCon","Duplex","TwnhsE","Twnhs",None],[10,40,70,100,130,0],inplace=True)
train["HouseStyle"].replace(["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl",None]
                            ,[10,30,50,70,90,110,130,150,0],inplace=True)
train["MasVnrType"].replace(["BrkCmn","BrkFace","CBlock","Stone","None",None],[1,1,1,1,0,0],inplace=True)
train["ExterQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["ExterCond"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["Foundation"].replace(["BrkTil","CBlock","PConc","Slab","Stone","Wood",None],[1,1,1,0,0,0,0],inplace=True)
train["BsmtQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["BsmtCond"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["Heating"].replace(["Floor","GasA","GasW","Grav","OthW","Wall",None],[0,1,0,0,0,0,0],inplace=True)
train["HeatingQC"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["CentralAir"].replace(["N","Y",None],[0,1,0],inplace=True)
train["Electrical"].replace(["SBrkr","FuseA","FuseF","FuseP","Mix",None],[1,0,0,0,0,0],inplace=True)
train["KitchenQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["Functional"].replace(["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal",None],[80,70,60,50,30,20,10,40,0],inplace=True)
train["FireplaceQu"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["GarageFinish"].replace(["Fin","RFn","Unf",None],[50,30,10,0],inplace=True)
train["GarageQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["GarageCond"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
train["PavedDrive"].replace(["Y","P","N",None],[20,10,0,0],inplace=True)
train["PoolQC"].replace(["Ex","Gd","TA","Fa",None],[40,30,20,10,0],inplace=True)
train["Fence"].replace(["GdPrv","MnPrv","GdWo","MnWw",None],[40,30,20,10,0],inplace=True)
#train["MiscFeature"] = train["MiscFeature"].replace(['Elev','Gar2','Othr','Shed','TenC',None],[1,1,1,1,1,0])
train["SaleCondition"].replace(["Normal","Abnorml","AdjLand","Alloca","Family","Partial",None],[1,0,0,0,0,0,0],inplace=True)

train["LotFrontage"]=train["LotFrontage"].fillna(train["LotFrontage"].mean())
train["MasVnrArea"]=train["MasVnrArea"].fillna(train["MasVnrArea"].mean())

train.drop(["Id","Alley","LotShape","LotConfig","Neighborhood","Condition1","Condition2","RoofStyle","RoofMatl",
                  "Exterior1st","Exterior2nd","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt",
                  "GarageArea","MoSold","MiscFeature","YrSold","SaleType","SalePrice"],axis=1,inplace=True)


X_train=train.iloc[:,1:].values


test["MSZoning"].replace(["A","C (all)","FV","I","RH","RL","RP","RM",None],[40,60,50,30,70,90,100,80,0],inplace=True)
test["Street"].replace(["Pave","Grvl",None],[100,50,0],inplace=True)
test["LandContour"].replace(["Lvl","Bnk","HLS","Low",None],[100,75,50,20,0],inplace=True)
test["Utilities"].replace(["AllPub","NoSewr","NoSeWa","ELO",None],[100,60,30,10,0],inplace=True)
test["LandSlope"].replace(["Gtl","Mod","Sev",None],[100,60,20,0],inplace=True)
test["BldgType"].replace(["1Fam","2fmCon","Duplex","TwnhsE","Twnhs",None],[10,40,70,100,130,0],inplace=True)
test["HouseStyle"].replace(["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl",None]
                            ,[10,30,50,70,90,110,130,150,0],inplace=True)
test["MasVnrType"].replace(["BrkCmn","BrkFace","CBlock","Stone","None",None],[1,1,1,1,0,0],inplace=True)
test["ExterQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["ExterCond"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["Foundation"].replace(["BrkTil","CBlock","PConc","Slab","Stone","Wood",None],[1,1,1,0,0,0,0],inplace=True)
test["BsmtQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["BsmtCond"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["Heating"].replace(["Floor","GasA","GasW","Grav","OthW","Wall",None],[0,1,0,0,0,0,0],inplace=True)
test["HeatingQC"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["CentralAir"].replace(["N","Y",None],[0,1,0],inplace=True)
test["Electrical"].replace(["SBrkr","FuseA","FuseF","FuseP","Mix",None],[1,0,0,0,0,0],inplace=True)
test["KitchenQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["Functional"].replace(["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal",None],[80,70,60,50,30,20,10,40,0],inplace=True)
test["FireplaceQu"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["GarageFinish"].replace(["Fin","RFn","Unf",None],[50,30,10,0],inplace=True)
test["GarageQual"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["GarageCond"].replace(["Ex","Gd","TA","Fa","Po",None],[50,40,30,20,10,0],inplace=True)
test["PavedDrive"].replace(["Y","P","N",None],[20,10,0,0],inplace=True)
test["PoolQC"].replace(["Ex","Gd","TA","Fa",None],[40,30,20,10,0],inplace=True)
test["Fence"].replace(["GdPrv","MnPrv","GdWo","MnWw",None],[40,30,20,10,0],inplace=True)
#test["MiscFeature"] = test["MiscFeature"].replace(['Elev','Gar2','Othr','Shed','TenC',None],[1,1,1,1,1,0])
test["SaleCondition"].replace(["Normal","Abnorml","AdjLand","Alloca","Family","Partial",None],[1,0,0,0,0,0,0],inplace=True)


test["BsmtUnfSF"]=test["BsmtUnfSF"].fillna(test["BsmtUnfSF"].mean())
test["TotalBsmtSF"]=test["TotalBsmtSF"].fillna(test["TotalBsmtSF"].mean())
test["BsmtFullBath"]=test["BsmtFullBath"].fillna(test["BsmtFullBath"].mean())
test["BsmtHalfBath"]=test["BsmtHalfBath"].fillna(test["BsmtHalfBath"].mean())





test.drop(["Id","Alley","LotShape","LotConfig","Neighborhood","Condition1","Condition2","RoofStyle","RoofMatl",
                  "Exterior1st","Exterior2nd","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt",
                  "GarageArea","MoSold","MiscFeature","YrSold","SaleType"],axis=1,inplace=True)


#coloumns having empty in test LotFrontage MasVnrArea   

test["LotFrontage"]=test["LotFrontage"].fillna(test["LotFrontage"].mean())
test["MasVnrArea"]=test["MasVnrArea"].fillna(test["MasVnrArea"].mean())
test["BsmtFinSF1"]=test["BsmtFinSF1"].fillna(test["BsmtFinSF1"].mean())
test["GarageCars"]=test["GarageCars"].fillna(test["GarageCars"].mean())
test["BsmtFinSF2"]=test["BsmtFinSF2"].fillna(test["BsmtFinSF2"].mean())




X_test=test.iloc[:,1:].values


model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,target)
prediction=model.predict(X_test)

user_id=np.arange(1461,2920)
data={'Id':user_id,
     'SalePrice':prediction}
a=pd.DataFrame(data)
a.to_csv("C:/Users/dell/Desktop/output.csv",index=False)

