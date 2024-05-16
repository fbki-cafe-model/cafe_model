from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_percentage_error
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
	print("Score:", model.score(test_x, test_y) )

	preds = model.predict(x)

	print("MAE:", mean_absolute_error(y, preds))

	# Visuals
	fig, axes = plt.subplots()

	# Ground truth
	axes.step(df.DMY, df["2to5"])
	axes.step(df.DMY, preds)

	axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
	fig.autofmt_xdate()

	plt.show()

class Agent():
	def __init__(self):
		self.planned_capacity = [400]*7
		self.history_demand = []
		self.history_plan = []
		self.history_met = []
		self.history_ts = []

		self.fig = None
		self.axes = None

	def visual_init(self):
		self.fig, self.axes = plt.subplots()

	def advance(self, ts, demand):
		plan = self.planned_capacity.pop(0)
		leftovers = plan - demand

		self.history_demand.append(demand)
		self.history_plan.append(plan)
		self.history_ts.append(ts)

		if leftovers >= 0:
			self.history_met.append(demand)
		else:
			self.history_met.append(demand + leftovers)

		if not self.planned_capacity:
			self.plan()
		else:
			if leftovers > 0:
				self.planned_capacity[0] += leftovers

	def plan(self):
		#self.planned_capacity = [400]*7
		self.planned_capacity = self.history_demand[-1:]
		#self.planned_capacity = self.history_demand[-7:]

	def visualize(self):
		if not self.fig:
			self.visual_init()

		self.axes.clear()
		self.axes.set_xlim(self.history_ts[0], self.history_ts[-1].ceil("30D"))
		self.axes.step(self.history_ts, self.history_demand)
		self.axes.step(self.history_ts, self.history_met)
		self.axes.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d (%a)"))
		self.fig.autofmt_xdate()
		plt.pause(.01)

	def score(self):
		print("MAPE:", mean_absolute_percentage_error(self.history_demand, self.history_plan))

def play_game():
	agent = Agent()

	time = np.float64(df.DMY)
	demand = df["2to5"]

	for ts, d in zip(df.DMY, demand):
		agent.advance(ts, d)
		agent.visualize()

	agent.score()

play_game()
