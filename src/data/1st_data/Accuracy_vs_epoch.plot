set terminal png

set output "test.png"
set datafile separator ','
#set style data line

set key off

#set format x "%.0s^%T"
set xrange [0:280]
set yrange [0:0.5]
#set xtics 10000,25000,240000
#set xtics rotate by 40 right

set title 'Accuracy vs. Num Epochs'
set xlabel 'Num Epochs'
set ylabel 'Accuracy'


plot 'plotting.csv' using 1:3 with lines
