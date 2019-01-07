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

import java.io.*;

public class ZMERT
{
  public static void main(String[] args) throws Exception
  {
    boolean external = false; // should each MERT iteration be launched externally?

    if (args.length == 1) {
      if (args[0].equals("-h")) { printZMERTUsage(args.length,true); System.exit(2); }
      else { external = false; }
    } else if (args.length == 3) { external = true; }
    else { printZMERTUsage(args.length,false); System.exit(1); }

    if (!external) {
      MertCore myMert = new MertCore(args[0]);
      myMert.run_MERT(); // optimize lambda[]!!!
      myMert.finish();
    } else {
      int maxMem = Integer.parseInt(args[1]);
      String configFileName = args[2];
      String stateFileName = "ZMERT.temp.state";
      String cp = System.getProperty("java.class.path");
      boolean done = false;
      int iteration = 0;
      while (!done) {
        ++iteration;
        Runtime rt = Runtime.getRuntime();
        Process p = rt.exec("java -Xmx" + maxMem + "m -cp " + cp + " MertCore " + configFileName + " " + stateFileName + " " + iteration);
        BufferedReader br_i = new BufferedReader(new InputStreamReader(p.getInputStream()));
        BufferedReader br_e = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        String dummy_line = null;
        while ((dummy_line = br_i.readLine()) != null) { System.out.println(dummy_line); }
        while ((dummy_line = br_e.readLine()) != null) { System.out.println(dummy_line); }
        int status = p.waitFor();

        if (status == 90) { done = true; }
        else if (status == 91) { done = false; }
        else { System.out.println("Z-MERT exiting prematurely (MertCore returned " + status + ")..."); break; }
      }
    }

    System.exit(0);

  } // main(String[] args)

  public static void printZMERTUsage(int argsLen, boolean detailed)
  {
    if (!detailed) {
      println("Oops, you provided " + argsLen + " args!");
      println("");
      println("Usage:");
      println("           ZMERT -maxMem maxMemoryInMB MERT_configFile");
      println("");
      println("Where -maxMem specifies the maximum amount of memory (in MB) Z-MERT is");
      println("allowed to use when performing its calculations (no memroy is needed while");
      println("the decoder is running),");
      println("and the config file contains any subset of Z-MERT's 20-some parameters,");
      println("one per line.  Run   ZMERT -h   for more details on those parameters.");
    } else {
      println("Usage:");
      println("           ZMERT -maxMem maxMemoryInMB MERT_configFile");
      println("");
      println("Where -maxMem specifies the maximum amount of memory (in MB) Z-MERT is");
      println("allowed to use when performing its calculations (no memroy is needed while");
      println("the decoder is running),");
      println("and the config file contains any subset of Z-MERT's 20-some parameters,");
      println("one per line.  Those parameters, and their default values, are:");
      println("");
      println("Relevant files:");
      println("  -dir dirPrefix: working directory\n    [[default: null string (i.e. they are in the current directory)]]");
      println("  -r refFile: target sentences (reference translations) of the MERT dataset\n    [[default: reference.txt]]");
      println("  -rps refsPerSen: number of reference translations per sentence\n    [[default: 1]]");
      println("  -txtNrm textNormMethod: how should text be normalized?\n       (0) don't normalize text,\n    or (1) \"NIST-style\", and also rejoin 're, *'s, n't, etc,\n    or (2) apply 1 and also rejoin dashes between letters,\n    or (3) apply 1 and also drop non-ASCII characters,\n    or (4) apply 1+2+3\n    [[default: 1]]");
      println("  -p paramsFile: file containing parameter names, initial values, and ranges\n    [[default: params.txt]]");
      println("  -docInfo documentInfoFile: file informing Z-MERT which document each\n    sentence belongs to\n    [[default: null string (i.e. all sentences are in one 'document')]]");
      println("  -fin finalLambda: file name for final lambda[] values\n    [[default: null string (i.e. no such file will be created)]]");
      println("");
      println("MERT specs:");
      println("  -m metricName metric options: name of evaluation metric and its options\n    [[default: BLEU 4 closest]]");
      println("  -maxIt maxMERTIts: maximum number of MERT iterations\n    [[default: 20]]");
      println("  -prevIt prevMERTIts: maximum number of previous MERT iterations to\n    construct candidate sets from\n    [[default: 20]]");
      println("  -minIt minMERTIts: number of iterations before considering an early exit\n    [[default: 5]]");
      println("  -stopIt stopMinIts: some early stopping criterion must be satisfied in\n    stopMinIts *consecutive* iterations before an early exit\n    [[default: 3]]");
      println("  -stopSig sigValue: early MERT exit if no weight changes by more than sigValue\n    [[default: -1 (i.e. this criterion is never investigated)]]");
      println("  -thrCnt threadCount: number of threads to run in parallel when optimizing\n    [[default: 1]]");
      println("  -save saveInter: save intermediate cfg files (1) or decoder outputs (2)\n    or both (3) or neither (0)\n    [[default: 3]]");
      println("  -compress compressFiles: should Z-MERT compress the files it produces (1)\n    or not (0)\n    [[default: 0]]");
      println("  -ipi initsPerIt: number of intermediate initial points per iteration\n    [[default: 20]]");
      println("  -opi oncePerIt: modify a parameter only once per iteration (1) or not (0)\n    [[default: 0]]");
      println("  -rand randInit: choose initial point randomly (1) or from paramsFile (0)\n    [[default: 0]]");
      println("  -seed seed: seed used to initialize random number generator\n    [[default: time (i.e. value returned by System.currentTimeMillis()]]");
//      println("  -ud useDisk: reliance on disk (0-2; higher value => more reliance)\n    [[default: 2]]");
      println("");
      println("Decoder specs:");
      println("  -cmd commandFile: name of file containing commands to run the decoder\n    [[default: ./dec_cmd]]");
      println("  -passIt passIterationToDecoder: should iteration number be passed\n    to command file (1) or not (0)\n    [[default: 0]]");
      println("  -decOut decoderOutFile: name of the output file produced by the decoder\n    [[default: output.nbest]]");
      println("  -decExit validExit: value returned by decoder to indicate success\n    [[default: 0]]");
      println("  -dcfg decConfigFile: name of decoder config file\n    [[default: dec_cfg.txt]]");
      println("  -N N: size of N-best list (per sentence) generated in each MERT iteration\n    [[default: 100]]");
      println("");
      println("Output specs:");
      println("  -v verbosity: Z-MERT verbosity level (0-2; higher value => more verbose)\n    [[default: 1]]");
      println("  -decV decVerbosity: should decoder output be printed (1) or ignored (0)\n    [[default: 0]]");
      println("");
    }
  }

  private static void println(Object obj) { System.out.println(obj); }

}
