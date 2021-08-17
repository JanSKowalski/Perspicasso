set terminal png

set output "test.png"
set datafile separator ','
#set style data line

set key off

#set format x "%.0s^%T"
set xrange [0:200]
set yrange [0:400]
set zrange [0:1]
#set xtics 10000,25000,240000
#set xtics rotate by 40 right

set title 'Accuracy vs. Num Epochs vs. Frame Size'
set xlabel 'Num Epochs'
set ylabel 'Frame Size'
set zlabel 'Accuracy'


splot 'plotting.csv' using 1:2:3 with lines
