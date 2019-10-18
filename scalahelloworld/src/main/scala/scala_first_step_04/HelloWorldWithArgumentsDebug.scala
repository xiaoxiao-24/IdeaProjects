package scala_first_step_04

import scala_first_step_03.HelloWorldWithArguments.args

object HelloWorldWithArgumentsDebug extends App {

  println("Hello World with arguments. ")

  println("Command line arguments are: ")
  println(args.mkString(", "))

}
