from pyspark.sql import SparkSession
import subprocess
import sys
import argparse
import json

subprocess.check_call([sys.executable, "-m", "pip", "install", "pydeequ==0.1.7"])

import pydeequ
from pydeequ.suggestions import *
from pydeequ.repository import FileSystemMetricsRepository, ResultKey
from pydeequ.checks import *
from pydeequ.verification import VerificationSuite


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_input_data", type=str)
    parser.add_argument("--s3_output_metrics_repository", type=str)
    args = parser.parse_args()
    print(f"Received arguments: {args}")

    spark = (
        SparkSession.builder.config("spark.jars.packages", pydeequ.deequ_maven_coord)
        .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
        .getOrCreate()
    )
    df = spark.read.csv(args.s3_input_data, header=True, inferSchema=True, sep="\t")

    try:
        suggestions = json.load(open("opt/ml/processing/input/suggested_constraints.json"))
    except Exception:
        suggestions = (
            ConstraintSuggestionRunner(spark)
            .onData(df)
            .addConstraintRule(CategoricalRangeRule())
            .addConstraintRule(CompleteIfCompleteRule())
            .addConstraintRule(NonNegativeNumbersRule())
            .addConstraintRule(RetainTypeRule())
            .addConstraintRule(RetainCompletenessRule())
            .run()
        )
        json.dump(suggestions, open("opt/ml/processing/output/suggestions/suggested_constraints.json", "w"))

    checks = Check(spark, CheckLevel.Warning, description="Suggested constraints")
    checks.addConstraints([eval("checks" + ct["code_for_constraint"]) for ct in suggestions["constraint_suggestions"]])

    repository = FileSystemMetricsRepository(spark, args.s3_output_metrics_repository)
    result_key = ResultKey(spark, ResultKey.current_milli_time())

    anls = (
        VerificationSuite(spark)
        .onData(df)
        .addCheck(checks)
        .useRepository(repository)
        .saveOrAppendResult(result_key)
        .run()
    )
