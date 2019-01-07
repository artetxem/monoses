import java.math.*;
import java.util.*;
import java.io.*;

public class Monoses extends EvaluationMetric {
  
  private static final int INTS_PER_DOUBLE = 8;
  private static final int BITS_PER_INT = 16;
  private static final int START_DIV = 16;
  private static final int NBLEU = 2;

  private BLEU bleu;
  private int suffStatsCountBleu;
  private int suffStatsCountLm;
  private double targetLMScore;

  /*
     You already have access to these data members of the parent
     class (EvaluationMetric):
         int numSentences;
           number of sentences in the MERT set
         int refsPerSen;
           number of references per sentence
         String[][] refSentences;
           refSentences[i][r] stores the r'th reference of the i'th
           source sentence (both indices are 0-based)
  */


  public Monoses(String[] options) {
    bleu = new BLEU(options);
    targetLMScore = Double.parseDouble(options[2]);
    initialize(); // set the data members of the metric
  }

  protected void initialize() {
    metricName = "monoses";
    toBeMinimized = false;
    suffStatsCountBleu = bleu.get_suffStatsCount();
    suffStatsCountLm = INTS_PER_DOUBLE + 1;
    suffStatsCount = NBLEU*suffStatsCountBleu + suffStatsCountLm;
  }

  public double bestPossibleScore() {
    return 1000000000;
  }

  public double worstPossibleScore() {
    return -1000000000;
  }

  public int[] suffStats(String output, int i) {
    final String fields[] = output.split("\t", 5);
    final String id = fields[0];
    final int direction = Integer.parseInt(fields[1]);
    final String translation = fields[2];
    final String candidate = fields[4].equals("-") ? null : fields[4];
    final double lm = candidate == null ? 0 : Double.parseDouble(fields[3]);
    final int stats[] = new int[suffStatsCount];
    if (candidate != null) {
      final int wordCount = candidate.trim().split("\\s+").length + 1;
      final int ints[] = doubleToInts(lm);
      for (int j = 0; j < INTS_PER_DOUBLE; j++) stats[NBLEU*suffStatsCountBleu+j] = ints[j];
      stats[suffStatsCount-1] = wordCount;
    }
    final int bleuStats[] = bleu.suffStats(translation, i);
    for (int j = 0; j < suffStatsCountBleu; j++) stats[direction*suffStatsCountBleu+j] = bleuStats[j];
    return stats;
  }

  public double score(int[] stats) {
    final int lmInts[] = new int[INTS_PER_DOUBLE];
    final int lmCount = stats[suffStatsCount-1];
    for (int i = 0; i < INTS_PER_DOUBLE; i++) lmInts[i] = stats[NBLEU*suffStatsCountBleu+i];
    double lmScore = lmScore(intsToDouble(lmInts), lmCount);

    double totalBleu = 0;
    double ratio = 1;
    final int bleuStats[] = new int[suffStatsCountBleu];
    for (int dir = 0; dir < NBLEU; dir++) {
      for (int i = 0; i < suffStatsCountBleu; i++) bleuStats[i] = stats[dir*suffStatsCountBleu+i];
      totalBleu += bleu.score(bleuStats);
      double c_len = bleuStats[suffStatsCountBleu-2];
      double r_len = bleuStats[suffStatsCountBleu-1];
      if (c_len > r_len) ratio *= c_len / r_len;
    }
    
    lmScore *= ratio;
    double lmPenalization = 0;
    if (lmScore > targetLMScore)
      lmPenalization = (lmScore-targetLMScore) * (lmScore-targetLMScore);
    return totalBleu - lmPenalization;
  }

  private double lmScore(double totalLog, int wordCount) {
    if (wordCount == 0) return 0;
    return - totalLog / wordCount;
  }

  public void printDetailedScore_fromStats(int[] stats, boolean oneLiner) {
    bleu.printDetailedScore_fromStats(stats, oneLiner);
  }

  private static int[] doubleToInts(double d) {
    final int ans[] = new int[INTS_PER_DOUBLE];
    for (int i = 0; i < INTS_PER_DOUBLE; i++) {
      final int exp = START_DIV - i*BITS_PER_INT;
      long mul = 1;
      long div = 1;
      if (exp >= 0) {
        div <<= exp;
      } else {
        mul <<= -exp;
      }
      ans[i] = (int)(mul*d/div);
      d -= div * ((double)ans[i]) / mul;
    }
    return ans;
  }

  private static double intsToDouble(int[] ints) {
    double ans = 0;
    for (int i = 0; i < INTS_PER_DOUBLE; i++) {
      final int exp = START_DIV - i*BITS_PER_INT;
      long mul = 1;
      long div = 1;
      if (exp >= 0) {
        div <<= exp;
      } else {
        mul <<= -exp;
      }
      ans += div * ((double)ints[i]) / mul;
    }
    return ans;
  }

}
