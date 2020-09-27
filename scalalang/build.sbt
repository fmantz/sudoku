enablePlugins(ScalaNativePlugin)

// The simplest possible sbt build file is just one line:
scalaVersion := "2.11.12"
// That is, to create a valid sbt build, all you've got to do is define the
// version of Scala you'd like your project to use.

// ============================================================================

// Lines like the above defining `scalaVersion` are called "settings" Settings
// are key/value pairs. In the case of `scalaVersion`, the key is "scalaVersion"
// and the value is "2.12.8"

scalacOptions ++= Seq("-deprecation", "-feature")

// It's possible to define many kinds of settings, such as:
name := "sudoku"
organization := "de.fmantz"
version := "0.1"

// You can define other libraries as dependencies in your build like this:
libraryDependencies += "org.scalactic" %% "scalactic" % "3.1.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.0" % "test"

fork := true
javaOptions in Compile += s"-DuniqueLibraryNames=true"

mainClass in assembly := Some("de.fmantz.sudoku.SudokuSolver")

nativeMode:="release"
