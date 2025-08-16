import argparse
from ensemble_inference import load_segmentator
from spaceai.benchmark.esa_competition_predictor import ESACompetitionPredictor
from spaceai.data import ESAMissions

def main(input_dir):
    test_parquet = f"{input_dir}/test.parquet"
    mission = ESAMissions.MISSION_1.value  # modifica se necessario
    artifacts_dir = "/app/experiments/<run_id>"  # sostituisci <run_id>
    segmentator = load_segmentator(artifacts_dir)
    predictor = ESACompetitionPredictor(artifacts_dir, segmentator, data_root=input_dir, mission=mission)
    submission = predictor.run(mission, test_parquet=test_parquet)
    submission[['id','is_anomaly']].to_csv(f"{input_dir}/submission.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="cartella contenente test.parquet")
    args = parser.parse_args()
    main(args.input_dir)
