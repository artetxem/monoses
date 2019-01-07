/* This file is part of the Z-MERT Training Tool for MT systems.
 * 
 * Z-MERT is an open-source tool, licensed under the terms of the
 * GNU Lesser General Public License (LGPL). Therefore, it is free
 * for personal and scientific use by individuals and/or research
 * groups. It may not be modified or redistributed, publicly or
 * privately, unless the licensing terms are observed. If in doubt,
 * contact the author for clarification and/or an explicit
 * permission.
 *
 * If you use Z-MERT in your work, please cite the following paper:
 *
 *       Omar F. Zaidan. 2009. Z-MERT: A Fully Configurable Open
 *       Source Tool for Minimum Error Rate Training of Machine
 *       Translation Systems. The Prague Bulletin of Mathematical
 *       Linguistics, No. 91:79-88.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307 USA
 */

import java.math.*;
import java.util.*;
import java.io.*;

public class TERMinusBLEU extends EvaluationMetric
{
  // individual components
  protected TER myTER;
  protected BLEU myBLEU;
  protected int suffStatsCount_TER;
  protected int suffStatsCount_BLEU;

  public TERMinusBLEU() { }

  public TERMinusBLEU(String[] Metric_options)
  {
    // M_o[0]: case sensitivity, case/nocase
    // M_o[1]: with-punctuation, punc/nopunc
    // M_o[2]: beam width, positive integer
    // M_o[3]: maximum shift distance, positive integer
    // M_o[4]: filename of tercom jar file
    // M_o[5]: number of threads to use for TER scoring (= number of tercom processes launched)
    // M_o[6]: maximum gram length, positive integer
    // M_o[7]: effective length calculation method, closest/shortest/average

    // for 0-3, default values in tercom-0.7.25 are: nocase, punc, 20, 50

    myTER = new TER(Metric_options);
    myBLEU = new BLEU(Integer.parseInt(Metric_options[6]),Metric_options[7]);

    initialize(); // set the data members of the metric
  }

  protected void initialize()
  {
    metricName = "TER-BLEU";
    toBeMinimized = true;
    suffStatsCount_TER = myTER.get_suffStatsCount();
    suffStatsCount_BLEU = myBLEU.get_suffStatsCount();
    suffStatsCount = suffStatsCount_TER + suffStatsCount_BLEU;
  }

  public double bestPossibleScore() { return -1.0; }
  public double worstPossibleScore() { return (+1.0 / 0.0); }

  public int[] suffStats(String cand_str, int i)
  {
    // this method should never be used when the metric is TER-BLEU,
    // because TERMinusBLEU.java overrides suffStats(String[],int[]) below,
    // which is the only method that calls suffStats(Sting,int).
    return null;
  }

  public int[][] suffStats(String[] cand_strings, int[] cand_indices)
  {
    // calculate sufficient statistics for each sentence in an arbitrary set of candidates

    int candCount = cand_strings.length;
    if (cand_indices.length != candCount) {
      System.out.println("Array lengths mismatch in suffStats(String[],int[]); returning null.");
      return null;
    }

    int[][] stats = new int[candCount][suffStatsCount];
      // size candCount x suffStatsCount
      //    = candCount x (suffStatsCount_TER + suffStatsCount_BLEU)

    int[][] stats_TER = myTER.suffStats(cand_strings,cand_indices);
      // size candCount x suffStatsCount_TER
    int[][] stats_BLEU = myBLEU.suffStats(cand_strings,cand_indices);
      // size candCount x suffStatsCount_BLEU

    for (int d = 0; d < candCount; ++d) {
      int s = 0;
      for (int s_T = 0; s_T < suffStatsCount_TER; ++s_T) {
        stats[d][s] = stats_TER[d][s_T];
        ++s;
      }

      for (int s_B = 0; s_B < suffStatsCount_BLEU; ++s_B) {
        stats[d][s] = stats_BLEU[d][s_B];
        ++s;
      }
    }

    return stats;

  }

  public void createSuffStatsFile(String cand_strings_fileName, String cand_indices_fileName, String outputFileName, int maxBatchSize)
  {
    try {
      myTER.createSuffStatsFile(cand_strings_fileName, cand_indices_fileName, outputFileName+".TER", maxBatchSize);
      myBLEU.createSuffStatsFile(cand_strings_fileName, cand_indices_fileName, outputFileName+".BLEU", maxBatchSize);

      PrintWriter outFile = new PrintWriter(outputFileName);

      FileInputStream inStream_TER = new FileInputStream(outputFileName+".TER");
      BufferedReader inFile_TER = new BufferedReader(new InputStreamReader(inStream_TER, "utf8"));

      FileInputStream inStream_BLEU = new FileInputStream(outputFileName+".BLEU");
      BufferedReader inFile_BLEU = new BufferedReader(new InputStreamReader(inStream_BLEU, "utf8"));

      String line_TER = inFile_TER.readLine();
      String line_BLEU = inFile_BLEU.readLine();

      // combine the two files into one
      while (line_TER != null) {
        outFile.println(line_TER + " " + line_BLEU);
        line_TER = inFile_TER.readLine();
        line_BLEU = inFile_BLEU.readLine();
      }

      inFile_TER.close();
      inFile_BLEU.close();
      outFile.close();

      File fd;
      fd = new File(outputFileName+".TER"); if (fd.exists()) fd.delete();
      fd = new File(outputFileName+".BLEU"); if (fd.exists()) fd.delete();
    } catch (IOException e) {
      System.err.println("IOException in TER.createTercomHypFile(...): " + e.getMessage());
      System.exit(99902);
    }
  }

  public double score(int[] stats)
  {
    if (stats.length != suffStatsCount) {
      System.out.println("Mismatch between stats.length and suffStatsCount (" + stats.length + " vs. " + suffStatsCount + ") in TERMinusBLEU.score(int[])");
      System.exit(1);
    }

    double sc = 0.0;

    int[] stats_TER = new int[suffStatsCount_TER];
    int[] stats_BLEU = new int[suffStatsCount_BLEU];
    for (int s = 0; s < suffStatsCount_TER; ++s) { stats_TER[s] = stats[s]; }
    for (int s = 0; s < suffStatsCount_BLEU; ++s) { stats_BLEU[s] = stats[s+suffStatsCount_TER]; }

    double sc_T = myTER.score(stats_TER);
    double sc_B = myBLEU.score(stats_BLEU);

    sc = sc_T - sc_B;

    return sc;
  }

  public void printDetailedScore_fromStats(int[] stats, boolean oneLiner)
  {
    int[] stats_TER = new int[suffStatsCount_TER];
    int[] stats_BLEU = new int[suffStatsCount_BLEU];
    for (int s = 0; s < suffStatsCount_TER; ++s) { stats_TER[s] = stats[s]; }
    for (int s = 0; s < suffStatsCount_BLEU; ++s) { stats_BLEU[s] = stats[s+suffStatsCount_TER]; }

    System.out.println("---TER---");
    myTER.printDetailedScore_fromStats(stats_TER, oneLiner);
    System.out.println("---BLEU---");
    myBLEU.printDetailedScore_fromStats(stats_BLEU, oneLiner);
    System.out.println("---------");
    System.out.println("  => " + metricName + " = " + f4.format(score(stats)));
  }

}

