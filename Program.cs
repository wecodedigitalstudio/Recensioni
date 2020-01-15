using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data; //per utilizzare IDataView
using Microsoft.ML.Runtime.Api; //per utilizzare ColumnName
using Microsoft.ML; //per usare questa libreria scaricare il pacchetto Microsoft ML
//tasto destro sulla solution, manage NuGet packages..., Browse, Microsoft.ML
namespace _20200114_MediaCelluleTrimestrePrecedente
{
    class FeedBackTrainingData
    {
        [Microsoft.ML.Runtime.Api.Column(ordinal: "0", name: "Label")] //sistemare tipo Column
        public bool IsGood { get; set; }
        [Microsoft.ML.Runtime.Api.Column(ordinal: "1")]
        public string FeedBackText { get; set; }
    }
    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }
    class Program
    {
        static List<FeedBackTrainingData> trainingdata =
            new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData =
            new List<FeedBackTrainingData>();
        static void LoadTestData() //dati di test
        {
            #region dati di test
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is good", //valore oltre il limite massimo
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = false
            });
            #endregion
        }
        static void LoadTrainingData() //dati di training
        {
            //sostituire poi con importazione file Excel
            #region dati di training
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shitty",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Average and ok",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "so nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad horrible",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "well ok ok",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "It very average",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "cool nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "sweet",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice and good",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "God horrible",
                IsGood = false
            });
            #endregion
        }
        static void Main(string[] args)
        {
            int i = 0;
            while (i<3)
            {
                i+=1;
                //1. carica i dati di training
                LoadTrainingData();

                //2. crea un oggetto di MLContext
                var mlContext = new MLContext();

                //3. converte i dato in IData View
                IDataView dataView = mlContext.CreateStreamingDataView<FeedBackTrainingData>(trainingdata);

                //4. crea la pipeline

                var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackText", "Features").Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

                //5. train
                var model = pipeline.Fit(dataView);

                //6. testare con dati appositi, diversi da quelli di training
                LoadTestData();
                IDataView dataView1 = mlContext.CreateStreamingDataView<FeedBackTrainingData>(testData);
                var predictions = model.Transform(dataView1);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
                Console.WriteLine(metrics.Accuracy);

                //7. utilizzare il modello
                Console.Write("Enter a feedback string: ");
                string feedbackstring = Console.ReadLine().ToString();
                var predictionFunction = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);
                var feedbackinput = new FeedBackTrainingData();
                feedbackinput.FeedBackText = feedbackstring;
                var feedbackpredicted = predictionFunction.Predict(feedbackinput);
                Console.WriteLine("Is good (predicted): " + feedbackpredicted.IsGood);
                Console.ReadLine();
            }            
        }
    }
}