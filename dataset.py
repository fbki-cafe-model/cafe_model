from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib.dates import AutoDateLocator, DateFormatter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("RestaurantDataVets_All_2to5.csv", na_values="?", parse_dates=[2])
df = df.dropna()

def filter():
	q_low = df["2to5"].quantile(0.01)
	q_hi  = df["2to5"].quantile(0.99)

	return df[(df["2to5"] < q_hi) & (df["2to5"] > q_low)]

df = filter()

"""
>>> df.dtypes
Index                       int64
Group                       int64
DMY                datetime64[ns]
MissingPrevDays             int64
Year                        int64
Day                         int64
January                     int64
February                    int64
March                       int64
April                       int64
May                         int64
June                        int64
July                        int64
August                      int64
September                   int64
October                     int64
November                    int64
December                    int64
Sunday                      int64
Monday                      int64
Tuesday                     int64
Wednesday                   int64
Thursday                    int64
Friday                      int64
Saturday                    int64
Holiday                     int64
Carnival                    int64
LentFasting                 int64
Ramadan                     int64
ChristmasSeason             int64
DailyAvg                  float64
WeeklyAvg                 float64
MinSales                  float64
MaxSales                  float64
DailyBusyness             float64
WeeklyBusyness            float64
2to5                      float64
dtype: object
"""

def overview():
	fig, axes = plt.subplots()

	axes.step(df.DMY, df["2to5"])

	axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
	fig.autofmt_xdate()

	plt.show()


# V1 pure linear with normalized time, seems to get decent score
# Notable:
# - Normalized time is important as apparent from ablations
# - Recovers average week of day visitors
def linear_v1():
	time = np.float64(df.DMY)[:, None]
	day_idx = time / 1000 / 1000/ 1000 / 60 / 60 / 24

	y = df["2to5"]
	x = np.hstack([
		time / time.max(),
		np.float64(df.Monday)[:, None],
		np.float64(df.Tuesday)[:, None],
		np.float64(df.Wednesday)[:, None],
		np.float64(df.Thursday)[:, None],
		np.float64(df.Friday)[:, None],
		np.float64(df.Saturday)[:, None],
		np.float64(df.Sunday)[:, None],
	])

	train_x, test_x = x[:800], x[800:]
	train_y, test_y = y[:800], y[800:]

	model = Lasso()
	model.fit(train_x, train_y)
	print( model.score(test_x, test_y) )

	preds = model.predict(x)

	# Visuals
	fig, axes = plt.subplots()

	# Ground truth
	axes.step(df.DMY, df["2to5"])
	axes.step(df.DMY, preds)

	axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
	fig.autofmt_xdate()

	plt.show()

# V2 with weekdays numbered in increasing order
# Notable:
# - Must not be a plain cumsum, only provides a linear increase if added as such
# - A better fit with sqrt
def linear_v2():
	time = np.float64(df.DMY)[:, None]
	day_idx = time / 1000 / 1000/ 1000 / 60 / 60 / 24

	y = df["2to5"]
	x = np.hstack([
		#time / time.max(),
		#np.float64(df.Monday)[:, None],
		#np.float64(df.Tuesday)[:, None],
		#np.float64(df.Wednesday)[:, None],
		#np.float64(df.Thursday)[:, None],
		#np.float64(df.Friday)[:, None],
		#np.float64(df.Saturday)[:, None],
		#np.float64(df.Sunday)[:, None],
		np.float64(df.Monday.cumsum() * df.Monday)[:, None] ** 0.3,
		np.float64(df.Tuesday.cumsum() * df.Tuesday)[:, None] ** 0.3,
		np.float64(df.Wednesday.cumsum() * df.Wednesday)[:, None] ** 0.3,
		np.float64(df.Thursday.cumsum() * df.Thursday)[:, None] ** 0.3,
		np.float64(df.Friday.cumsum() * df.Friday)[:, None] ** 0.3,
		np.float64(df.Saturday.cumsum() * df.Saturday)[:, None] ** 0.3,
		np.float64(df.Sunday.cumsum() * df.Sunday)[:, None] ** 0.3,
	])

	train_x, test_x = x[:800], x[800:]
	train_y, test_y = y[:800], y[800:]

	model = Lasso()
	model.fit(train_x, train_y)
	print( model.score(test_x, test_y) )

	preds = model.predict(x)

	# Visuals
	fig, axes = plt.subplots()

	# Ground truth
	axes.step(df.DMY, df["2to5"])
	axes.step(df.DMY, preds)

	axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
	fig.autofmt_xdate()

	plt.show()

linear_v2()
