#!/bin/bash

#
# sudoku - Sudoku solver for comparison Scala with Rust
#        - The motivation is explained in the README.md file in the top level folder.
# Copyright (C) 2020 Florian Mantz
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

export LEVEL=6 #CONFIG HOW MANY LEVELS ARE RUN HERE!!!
export NEWDIR="$(date '+%Y-%m-%d_%H-%M-%S').logs"

mkdir /root/performance/$NEWDIR

let NUMBER_OF_PUZZLES=1

for (( c=1; c<=LEVEL; c++ ))
do  

echo *****************************************
echo LEVEL = $c   PUZZLES = $NUMBER_OF_PUZZLES
echo *****************************************

#Generate sudokus with QQwing (if not already generated):
if [[ ! -f "/root/performance/sudoku${NUMBER_OF_PUZZLES}.txt" ]]
then
    echo "generate '/root/performance/sudoku${NUMBER_OF_PUZZLES}.txt'"
    /usr/bin/time -v -a --output=/root/performance/qqwing_gen.log -p sh -c "java -jar /root/qqwing-1.3.4.jar --generate ${NUMBER_OF_PUZZLES} --difficulty expert --compact > /root/performance/sudoku${NUMBER_OF_PUZZLES}.txt"
fi

#Solve sudokus with QQwing (reference):
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-qqwing.log -p sh -c "cat /root/performance/sudoku${NUMBER_OF_PUZZLES}.txt | java -jar /root/qqwing-1.3.4.jar --solve --compact > /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-qqwing.txt"

#Solve sudokus with rust sudoku:
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-rust.log -p sh -c "/root/sudoku-rust /root/performance/sudoku${NUMBER_OF_PUZZLES}.txt /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-rust.txt"

#Solve sudokus with scala sudoku:
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-scala.log -p sh -c "java -jar /root/sudoku-scala.jar /root/performance/sudoku${NUMBER_OF_PUZZLES}.txt /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-scala.txt"

#DISABLED: only useful for versions < 0.3 (!)
#Solve sudokus with scalanative sudoku:
#/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-scalanative.log -p sh -c "/root/sudoku-scalanative /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}.txt /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-scalanative.txt"

let NUMBER_OF_PUZZLES=$(($NUMBER_OF_PUZZLES*10))

#Update result csvs:
scala /root/performance/prepare_data.sh /root/performance/${NEWDIR}

done

exit
