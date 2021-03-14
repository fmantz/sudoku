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

lazy val isNative = sys.props.get("NATIVE").isDefined
lazy val enablePluginsList = if(isNative) Seq(ScalaNativePlugin) else Seq.empty
enablePlugins(enablePluginsList:_*)

scalaVersion := "2.11.12"

scalacOptions ++= Seq("-deprecation", "-feature")

// It's possible to define many kinds of settings, such as:
name := "sudoku"
organization := "de.fmantz"
version := "0.7.0"

// You can define other libraries as dependencies in your build like this:
libraryDependencies += "org.scalactic" %% "scalactic" % "3.1.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.0" % "test"

fork := true
javaOptions in Compile += s"-DuniqueLibraryNames=true"

mainClass in assembly := Some("de.fmantz.sudoku.SudokuSolver")

nativeMode:="release"
