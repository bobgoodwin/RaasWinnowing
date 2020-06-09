using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace WinnowLearn
{

    public interface IModelEntity
    {
        void PrintToConsole();
    }

    public class TrainingData : IModelEntity
    {
        public static int[] Range = new int[] { 25, 75, 175, 375, 775, 1575 };
        public float Label;
        public float DocCount;
        public float l1_0;
        public float l2_0;
        public float l1_1;
        public float l2_1;
        public float l1_2;
        public float l2_2;
        public float l1_3;
        public float l2_3;
        public float l1_4;
        public float l2_4;
        public float l1_5;
        public float l2_5;
        public int level;

        public void PrintToConsole()
        {
            Console.WriteLine($"Label: {Label}");
            Console.Write($"Features: [l1_0] {l1_0} [l2_0] {l2_0} ");
            Console.Write($"[DocCount] {DocCount} [l1_1] {l1_1} [l2_1] {l2_1} ");
            Console.Write($"[l1_2] {l1_2} [l2_2] {l2_2} ");
            Console.Write($"[l1_3] {l1_3} [l2_3] {l2_3} ");
            Console.Write($"[l1_4] {l1_4} [l2_4] {l2_4} ");
            Console.WriteLine($"[l1_5] {l1_5} [l2_5] {l2_5} ");
        }

        public TrainingData() { }
        public TrainingData(TrainingData b, int level)
        {
            this.Label = b.Label;
            this.DocCount = b.DocCount;
            this.l1_0 = b.l1_0;
            this.l2_0 = b.l2_0;
            this.l1_1 = b.l1_1;
            if (level >= 2)
                this.l2_1 = b.l2_1;
            this.l1_2 = b.l1_2;
            if (level >= 3)
                this.l2_2 = b.l2_2;
            this.l1_3 = b.l1_3;
            if (level >= 4)
                this.l2_3 = b.l2_3;
            this.l1_4 = b.l1_4;
            if (level >= 5)
                this.l2_4 = b.l2_4;
            this.l1_5 = b.l1_5;
            if (level >= 6)
                this.l2_5 = b.l2_5;
        }
    }

    public class Refined
    {
        public float[] threshold = new float[6];
    }

    public class RefineGroup
    {
        public float[][] group = new float[6][6];
    }

    public class RefineGroupByPct
    {
        public Dictionary<float, RefineGroup> dict = new Dictionary<float, RefineGroup>();
    }
    public class EvaluatedData
    {
        public int level;
        public float Label;
        public float DocCount;
        public float[] Score=new float[6];
    }

    public class Score2
    {
        public float Score;
    }


    class Program
    {
        static List<TrainingData> TrainingData0 = new List<TrainingData>();
        static List<TrainingData> TrainingData1 = new List<TrainingData>();
        static List<TrainingData> TrainingData2 = new List<TrainingData>();
        static void Main(string[] args)
        {
            ReadRawData();
            TrainAndEvaluate(TrainingData0, TrainingData1, TrainingData2);
        }

        static List<TrainingData>[,] Group(List<TrainingData> data)
        {
            // [1,1] = stop after 25, only have l1 scores for top 25
            // [1,2] = stop after 25, only have l1 scores for top 75
            // [1,3] = stop after 25, only have l1 scores for top 175
            // [1,4] = stop after 25, only have l1 scores for top 375
            // [1,5] = stop after 25, only have l1 scores for top 775
            // [1,6] = stop after 25,  have l1 scores for all 1575
            // [2,2] = stop after 75, only have l1 scores for top 75
            // [2,2] = stop after 75, only have l1 scores for top 175
            // ...
            // [7,7] = stop after 1775, have l1 score for all 1775
            List<TrainingData>[,] groups = new List<TrainingData>[7, 7];

            for (int stop = 1; stop <= 6; stop++)
            {
                for (int have = stop; have <= 6; have++)
                {
                    groups[stop, have] = new List<TrainingData>();
                }
            }

            foreach (var d in data)
            {
                for (int stop = 1; stop <= d.level; stop++)
                {
                    groups[stop, d.level].Add(new TrainingData(d, stop));
                }
            }

            return groups;
        }

        static void TrainAndEvaluate(List<TrainingData> trainData, List<TrainingData> refineData, List<TrainingData> testData)
        {
            Trained[,] trained = TrainGroups(trainData);
            RefineGroupByPct refinedGroupByPct = RefineGroupsAll(trained, refineData);
            TestGroupAll(trained, refinedGroupByPct, testData);
        }


        static Dictionary<float,float> TestGroupAll(Trained[,] trained, RefineGroupByPct refinedGroupByPct, List<TrainingData> testData)
        {
            Dictionary<float, float> result = new Dictionary<float, float>();
            foreach (var f in refinedGroupByPct.dict.Keys)
            {
                result[f] = TestGroup(trained, refinedGroupByPct.dict[f], testData);
            }

            return result;
        }
        static RefineGroupByPct RefineGroupsAll(Trained[,] trained, List<TrainingData> refineData)
        {
            RefineGroupByPct result = new RefineGroupByPct();
            for (float f = .999F; f >= .99; f -= .001F)
            {
                result.dict[f] = RefineGroup(trained, refineData, f);
            }

            for (float f = .99F; f >= .90; f -= .01F)
            {
                result.dict[f] = RefineGroup(trained, refineData, f);
            }

            for (float f = .90F; f >= .0; f -= .1F)
            {
                result.dict[f] = RefineGroup(trained, refineData, f);
            }

            return result;
        }


        static float TestGroup(Trained[,] trained, RefineGroup g, List<TrainingData> testData)
        {
            List<EvaluatedData>[] evaluations = Evaluations(trained, testData);
            float tot = 0;
            float cnt = 0;
            for (int have = 1; have <= 6; have++)
            {
                tot+=Test(evaluations[have], testData, g.group[have], have);
                cnt += evaluations[have].Count();
            }

            return tot / cnt;
        }
        static RefineGroup RefineGroup(Trained[,] trained, List<TrainingData> refineData, float pct)
        {
            RefineGroup result = new RefineGroup();
            result.group = new float[6][];

            List<EvaluatedData>[] evaluations = Evaluations(trained, refineData);

            for (int have = 1; have <= 6; have++)
            {
                result.group[have-1]=
                Refined(evaluations[have], refineData, pct, result, have);
            }

            return result;
        }

        static float Test(List<EvaluatedData> evaluations, List<TrainingData> refineData, float[] current, int have)
        {
            float totcost = 0;
            float totok = 0;
            float cnt = 0;
            foreach (var ev in evaluations)
            {
                bool ok = Score(ev, current, out float cost);
                totok += ok ? 1 : 0;
                totcost += cost;
                cnt++;
            }

            return totcost / cnt;
        }

        /// <summary>
        /// Solver finds best threshold for each of the 6 stop points in order to have the lowest cost for a given pct threshold.
        /// </summary>
        /// <param name="trained"></param>
        /// <param name="refineData"></param>
        /// <param name="pct"></param>
        /// <param name="refined"></param>
        static float[] Refined(List<EvaluatedData> evaluations, List<TrainingData> refineData, float pct, RefineGroup result, int have)
        {
            float[] min = new float[6];
            float[] max = new float[6];
            for (int i = 0; i < 6; i++)
            {
                min[i] = -10000;
                max[i] = 10000;
            }
            const int divisions = 10;
            float dist = ((float)max[0] - min[0]) / divisions;
            float[] best = null;
            float bestcost = float.MaxValue;
            while (dist > .1)
            {
                float[] current = new float[6];
                for (current[0] = min[0]; current[0] <= max[0]; current[0] += dist)
                    for (current[1] = min[1]; current[1] <= max[1]; current[1] += have >= 2 ? dist : float.MaxValue)
                        for (current[2] = min[2]; current[2] <= max[2]; current[2] += have >= 3 ? dist : float.MaxValue)
                            for (current[3] = min[3]; current[3] <= max[3]; current[3] += have >= 4 ? dist : float.MaxValue)
                                for (current[4] = min[4]; current[4] <= max[4]; current[4] += have >= 5 ? dist : float.MaxValue)
                                    for (current[5] = min[5]; current[5] <= max[5]; current[5] += have >= 6 ? dist : float.MaxValue)
                                    {
                                        float totcost=0;
                                        float totok = 0;
                                        float cnt = 0;
                                        foreach (var ev in evaluations)
                                        {
                                            bool ok = Score(ev, current, out float cost);
                                            totok += ok ? 1 : 0;
                                            totcost += cost;
                                            cnt++;
                                        }

                                        if (totok/cnt>=pct)
                                        {
                                            float avecost = totcost/cnt;
                                            if (avecost < bestcost)
                                            {
                                                bestcost = avecost;
                                                best = current;
                                            }
                                        }
                                    }


                dist = dist / 5;
                for (int i = 0; i<6; i++)
                {
                    min[i] = current[i] - dist * divisions / 2;
                    max[i] = current[i] + dist * divisions / 2;
                }
            }

            return current;
        }

        static bool Score(EvaluatedData ev, float[] current, out float cost)
        {
            float denom = ev.DocCount;
            float end = ev.DocCount;
            for (int i = 0; i < 6; i++)
            {
                if (ev.Score[i] <= current[i])
                {
                    end = TrainingData.Range[0];
                    break;
                }
            }

            cost = (float)end / denom;

            return end <= ev.Label;
        }

        static List<EvaluatedData>[] Evaluations(Trained[,] trained, List<TrainingData> refineData)
        {
            List<EvaluatedData>[] result = new List<EvaluatedData>[7];
            for (int have = 1; have <= 6; have++)
            {
                result[have] = new List<EvaluatedData>();


                result[have] = Evaluations(trained, refineData, have);
            }
            return result;
        }

        static List<EvaluatedData> Evaluations(Trained[,] trained, List<TrainingData> refineData, int have)
        {
            List<EvaluatedData> evaluations = new List<EvaluatedData>();
            foreach (var td in refineData)
            {
                var e = Evaluate(trained, td, have);
                if (e != null) evaluations.Add(e);
            }

            return evaluations;
        }

        static EvaluatedData Evaluate(Trained[,] trained, TrainingData refineData, int have)
        {
            if (refineData.level != have) return null;
            EvaluatedData e = new EvaluatedData() { level = refineData.level, Label = refineData.Label, DocCount = refineData.DocCount };
            for (int stop = 1; stop <= have; stop++)
            {
                e.Score[stop-1] = trained[stop,have].predictor.Predict(new TrainingData(refineData, stop)).Score;
            }


            return e;
        }

        struct Trained
        {
            public TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> model;
            public PredictionEngine<TrainingData, Score2> predictor;
        }

        static Trained[,] TrainGroups(List<TrainingData> trainData)
        {
            List<TrainingData>[,] groups = Group(trainData);
            Trained[,] trained = new Trained[7, 7];
            for (int stop = 1; stop <= 6; stop++)
            {
                for (int have = stop; have <= 6; have++)
                {
                    trained[stop, have] = TrainOne(groups[stop, have]);
                }
            }

            return trained;
        }

        static Trained TrainOne(List<TrainingData> trainData)
        {
            MLContext mlContext = new MLContext(seed: 0);
            IDataView baseTrainingDataView = mlContext.Data.LoadFromEnumerable<TrainingData>(trainData);
            var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TrainingData.Label))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.DocCount)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l1_0)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l2_0)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l1_1)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l2_1)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l1_2)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l2_2)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l1_3)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l2_3)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l1_4)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l2_4)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l1_5)))
                     .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TrainingData.l2_5)))
                     .Append(mlContext.Transforms.Concatenate("Features", "DocCount", "l1_0", "l2_0",
                     "l1_0", "l2_0",
                     "l1_0", "l2_0",
                     "l1_1", "l2_1",
                     "l1_1", "l2_1",
                     "l1_2", "l2_2",
                     "l1_2", "l2_2",
                     "l1_3", "l2_3",
                     "l1_3", "l2_3",
                     "l1_4", "l2_4",
                     "l1_4", "l2_4",
                     "l1_5", "l2_5",
                     "l1_5", "l2_5",
                     nameof(TrainingData.DocCount), nameof(TrainingData.l1_0), nameof(TrainingData.l2_0),
                     nameof(TrainingData.l1_1), nameof(TrainingData.l2_1),
                     nameof(TrainingData.l1_2), nameof(TrainingData.l2_2),
                     nameof(TrainingData.l1_3), nameof(TrainingData.l2_3),
                     nameof(TrainingData.l1_4), nameof(TrainingData.l2_4),
                     nameof(TrainingData.l1_5), nameof(TrainingData.l2_5)));
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            //The pipeline is trained on the dataset that has been loaded and transformed.
            Console.WriteLine("=============== Training the model ===============");
            TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainedModel = trainingPipeline.Fit(baseTrainingDataView);
            Console.WriteLine("=============== End of training process ===============");
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<TrainingData, Score2>(trainedModel);

            return new Trained()
            {
                model = trainedModel,
                predictor = predEngine
            };

        }

        static void ReadRawData()
        {
            using (TextReader tr = new StreamReader(@"raas.csv"))
            {
                string line;
                int cnt = 0;
                while (null != (line = tr.ReadLine()))
                {
                    string[] parts = line.Split('\t');
                    if (parts.Length >= 3)
                    {
                        cnt++;
                        TrainingData d = new TrainingData();
                        d.Label = float.Parse(parts[0]);
                        d.DocCount = float.Parse(parts[1]);
                        d.l1_0 = float.Parse(parts[2]);
                        d.l2_0 = float.Parse(parts[3]);
                        d.level = 1;

                        if (parts.Length >= 6)
                        {
                            d.l1_1 = float.Parse(parts[4]);
                            d.l2_1 = float.Parse(parts[5]);
                            d.level = 2;
                        }
                        else
                        {
                            d.l1_1 = -5;
                        }

                        if (parts.Length >= 8)
                        {
                            d.l1_2 = float.Parse(parts[6]);
                            d.l2_2 = float.Parse(parts[7]);
                            d.level = 3;
                        }
                        else
                        {
                            d.l1_2 = -5;
                        }

                        if (parts.Length >= 10)
                        {
                            d.l1_3 = float.Parse(parts[8]);
                            d.l2_3 = float.Parse(parts[9]);
                            d.level = 4;
                        }
                        else
                        {
                            d.l1_3 = -5;
                        }

                        if (parts.Length >= 12)
                        {
                            d.l1_4 = float.Parse(parts[10]);
                            d.l2_4 = float.Parse(parts[11]);
                            d.level = 5;
                        }
                        else
                        {
                            d.l1_4 = -5;
                        }

                        if (parts.Length >= 14)
                        {
                            d.l1_5 = float.Parse(parts[12]);
                            d.l2_5 = float.Parse(parts[13]);
                            d.level = 6;
                        }
                        else
                        {
                            d.l1_5 = -5;
                        }

                        if (cnt % 3 == 0)
                            TrainingData0.Add(d);
                        else if (cnt % 3 == 1)
                            TrainingData1.Add(d);
                        else if (cnt % 3 == 2)
                            TrainingData2.Add(d);
                    }
                }

            }
        }
    }
}
