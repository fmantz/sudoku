import mill._, scalalib._
import mill.scalalib.Assembly._

object sudoku extends RootModule with SbtModule {

	override def scalaVersion = "2.13.11"
	val name = "sudoku"
	val publishVersion = "0.9.0"

	override def mainClass = Some("de.fmantz.sudoku.SudokuSolver")

	override def ivyDeps = Agg(
		ivy"org.scala-lang.modules::scala-parallel-collections:1.0.0"
	)

	object test extends SbtModuleTests {
		override def ivyDeps = Agg(
			ivy"org.scalatest::scalatest:3.1.0"
		)

		override def testFramework = "org.scalatest.tools.Framework"
	}

	// add sources for testing from test/resources folder:
	override def resources = T {
		super.resources() :+ PathRef(millSourcePath / "src" / "test" / "resources")
	}

	override def assembly: T[PathRef] = T {
		val dest = T.dest / s"${name}-${publishVersion}-assembly.jar"
		os.copy(super.assembly().path, dest)
		PathRef(dest)
	}

	override def assemblyRules = Seq(
		Rule.ExcludePattern(".*\\.txt")
	)

}
