from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore") # 忽略警告
from sklearn.ensemble import RandomForestClassifier

# 读取数据
data = pd.read_excel('数据源.xlsx', sheet_name='Sheet1')

# 分离特征和标签
x = data.drop('Adverse outcome', axis=1)
y = data['Adverse outcome']

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)

# 模型训练
model = RandomForestClassifier(n_estimators=50, random_state=17)
model.fit(x_train, y_train)

# 保存训练好的XGBoost模型
joblib.dump(model, "RF.pkl")


