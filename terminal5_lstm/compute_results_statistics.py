import sys
import numpy as np

def read_results(filename):
	f = open(filename)
	header = f.readline()
	header = [x.strip() for x in header.split(",")]
	vals = [[] for x in range(len(header))]
	for line in f:
		splt = line.strip().split(",")
		for i in xrange(len(splt)-1):
			vals[i].append(float(splt[i]))
	f.close()
	vals = np.array(vals)
	return vals, header

mae_vals, header = read_results("results_mae.txt")
mae_means = np.mean(mae_vals, axis=1)
mae_stds = np.std(mae_vals, axis=1)
mae_best = np.min(mae_vals, axis=1)

rae_vals, header = read_results("results_rae.txt")
rae_means = np.mean(rae_vals, axis=1)
rae_stds = np.std(rae_vals, axis=1)
rae_best = np.min(rae_vals, axis=1)

rmse_vals, header = read_results("results_rmse.txt")
rmse_means = np.mean(rmse_vals, axis=1)
rmse_stds = np.std(rmse_vals, axis=1)
rmse_best = np.min(rmse_vals, axis=1)

rrse_vals, header = read_results("results_rrse.txt")
rrse_means = np.mean(rrse_vals, axis=1)
rrse_stds = np.std(rrse_vals, axis=1)
rrse_best = np.min(rrse_vals, axis=1)

mape_vals, header = read_results("results_mape.txt")
mape_means = np.mean(mape_vals, axis=1)
mape_stds = np.std(mape_vals, axis=1)
mape_best = np.min(mape_vals, axis=1)

r2_vals, header = read_results("results_r2.txt")
r2_means = np.mean(r2_vals, axis=1)
r2_stds = np.std(r2_vals, axis=1)
r2_best = np.max(r2_vals, axis=1)

print "Num runs:", mae_vals.shape[1]
#print "Results statistics:"

print "Method,MAE (mean),MAE (std),MAE (min),RAE (mean),RAE (std),RAE (min),RMSE (mean),RMSE (std),RMSE (min),RRSE (mean),RRSE (std),RRSE (min),MAPE (mean),MAPE (std),MAPE (min),R2 (mean),R2 (std),R2 (min)"
for i in xrange(len(header)):
	s = header[i]
	s += ",%.3f,%.3f,%.3f" % (mae_means[i],mae_stds[i],mae_best[i])
	s += ",%.3f,%.3f,%.3f" % (rae_means[i],rae_stds[i],rae_best[i])
	s += ",%.3f,%.3f,%.3f" % (rmse_means[i],rmse_stds[i],rmse_best[i])
	s += ",%.3f,%.3f,%.3f" % (rrse_means[i],rrse_stds[i],rrse_best[i])
	s += ",%.3f,%.3f,%.3f" % (mape_means[i],mape_stds[i],mape_best[i])
	s += ",%.3f,%.3f,%.3f" % (r2_means[i],r2_stds[i],r2_best[i])
	print s

print 
print "FOR LATEX:"
#print "Method & MAE & RAE & RMSE & RRSE  & MAPE & R2\\\\"
print "Method & MAE & RMSE & MAPE & R2\\\\"
header = ["LR L","LR L+W","LR L+W+E1","LR L+W+E1","LR L+W+E2","DL L","DL L+W","DL L+W+E1","DL L+W+E1","DL L+W+E2","DL L+W+E1+T","DL L+W+E2+T"]
for i in xrange(len(header)):
	s = header[i]
	s += " & %.1f ($\pm$%.1f)" % (mae_means[i],mae_stds[i])
	#s += " & %.3f ($\pm$%.3f)" % (rae_means[i],rae_stds[i])
	s += " & %.1f ($\pm$%.1f)" % (rmse_means[i],rmse_stds[i])
	#s += " & %.3f ($\pm$%.3f)" % (rrse_means[i],rrse_stds[i])
	s += " & %.1f ($\pm$%.1f)" % (mape_means[i],mape_stds[i])
	s += " & %.3f ($\pm$%.3f)" % (r2_means[i],r2_stds[i])
	print s+"\\\\"
