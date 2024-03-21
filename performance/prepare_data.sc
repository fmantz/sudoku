#!/usr/bin/env -S /root/.sdkman/candidates/scalacli/current/bin/scala-cli shebang


//scalastyle:off
/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.md file in the top level folder.
 * Copyright (C) 2020 Florian Mantz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
//scalastyle:on
import java.io.{BufferedWriter, File, FileWriter, PrintWriter}
import scala.collection.mutable
import scala.io.Source

val TimeString: String = "Elapsed (wall clock) time (h:mm:ss or m:ss):"
val MemString: String = "Maximum resident set size (kbytes):"

/**
 * Write csv file into
 */
def writeFile(file: File, csvData: Array[Array[String]]): Unit = {
  val bw = new PrintWriter(new BufferedWriter(new FileWriter(file)))
  csvData.foreach(row =>
    bw.println(row.mkString(","))
  )
  bw.close()
}

if (args.isEmpty) {
  sys.error("working directory not provided as argument!")
}

// println(s"Work on dir '${args.head}'")

val workingDir = new File(args.head)

val filesToConsider = workingDir
  .listFiles()
  .filter({
    f =>
      val name = f.getName
      name.startsWith("sudoku") && name.endsWith("log")
  })

val rsData = mutable.Map.empty[String, (Array[String], Array[String])]

//Extract data:
for (file <- filesToConsider) {

  val currentName = file.getName
  val currentProgram = currentName.substring(currentName.indexOf('-') + 1, currentName.indexOf('.'))
  val currentFile = Source.fromFile(file)

  val currentData = currentFile
    .getLines()
    .map(_.trim)
    .filter(line => line.trim.startsWith(TimeString) || line.trim.startsWith(MemString))
    .toArray

  val timeData = Array.ofDim[String](currentData.length / 2)
  val memData = Array.ofDim[String](currentData.length / 2)

  val timeStringLength = TimeString.length
  val memStringLength = MemString.length

  currentData.indices.foreach(i => {
    val rsIndex = i / 2
    i % 2 match {
      case 0 => timeData(rsIndex) = currentData(i).substring(timeStringLength).trim
      case 1 => memData(rsIndex) = currentData(i).substring(memStringLength).trim
    }
  })

  rsData.put(currentProgram, (timeData, memData))
}

val countPrograms: Int = rsData.size
val countLevels: Int = rsData.values.map(d => math.max(d._1.length, d._2.length)).max

val csvTime = Array.ofDim[Array[String]](countPrograms + 1)
val csvMem = Array.ofDim[Array[String]](countPrograms + 1)

val header: Array[String] = Array("#puzzles:") ++ Array.tabulate(countLevels)(i => math.round(math.pow(10, i)).toString)
csvTime(0) = header
csvMem(0) = header

var myIndex = 1
rsData
  .toArray
  .sortBy(_._1)
  .foreach({
    case (progName, (time, mem)) =>
      csvTime(myIndex) = Array(progName) ++ time
      csvMem(myIndex) = Array(progName) ++ mem
      myIndex += 1
  })

writeFile(new File(workingDir, "time.csv"), csvTime)
writeFile(new File(workingDir, "mem.csv"), csvMem)

println("data prepared!")
