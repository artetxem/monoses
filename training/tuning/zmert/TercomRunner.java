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

import java.util.*;
import java.io.*;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;

public class TercomRunner implements Runnable
{
  /* non-static data members */
  private Semaphore blocker;

  private String refFileName;
  private String hypFileName;
  private String outFileNamePrefix;
  private int memSize;

  /* static data members */
  private static boolean caseSensitive;
  private static boolean withPunctuation;
  private static int beamWidth;
  private static int maxShiftDist;
  private static String tercomJarFileName;

  public static void set_TercomParams(
      boolean in_caseSensitive, boolean in_withPunctuation, int in_beamWidth, int in_maxShiftDist, String in_tercomJarFileName)
  {
    caseSensitive = in_caseSensitive;
    withPunctuation = in_withPunctuation;
    beamWidth = in_beamWidth;
    maxShiftDist = in_maxShiftDist;
    tercomJarFileName = in_tercomJarFileName;
  }

  public TercomRunner(Semaphore in_blocker, String in_refFileName, String in_hypFileName, String in_outFileNamePrefix, int in_memSize)
  {
    blocker = in_blocker;
    refFileName = in_refFileName;
    hypFileName = in_hypFileName;
    outFileNamePrefix = in_outFileNamePrefix;
    memSize = in_memSize;
  }

  private void real_run() {

    try {

      String cmd_str = "java -Xmx" + memSize + "m -Dfile.encoding=utf8 -jar " + tercomJarFileName + " -r " + refFileName + " -h " + hypFileName + " -o ter -n " + outFileNamePrefix;
      cmd_str += " -b " + beamWidth;
      cmd_str += " -d " + maxShiftDist;
      if (caseSensitive) { cmd_str += " -s"; }
      if (!withPunctuation) { cmd_str += " -P"; }
      /* From tercom's README:
           -s case sensitivity, optional, default is insensitive
           -P no punctuations, default is with punctuations.
      */

      Runtime rt = Runtime.getRuntime();
      Process p = rt.exec(cmd_str);

      StreamGobbler errorGobbler = new StreamGobbler(p.getErrorStream(), 0);
      StreamGobbler outputGobbler = new StreamGobbler(p.getInputStream(), 0);

      errorGobbler.start();
      outputGobbler.start();

      int exitValue = p.waitFor();

      File fd;
      fd = new File(hypFileName); if (fd.exists()) fd.delete();
      fd = new File(refFileName); if (fd.exists()) fd.delete();

    } catch (IOException e) {
      System.err.println("IOException in TER.runTercom(...): " + e.getMessage());
      System.exit(99902);
    } catch (InterruptedException e) {
      System.err.println("InterruptedException in TER.runTercom(...): " + e.getMessage());
      System.exit(99903);
    }

    blocker.release();

  }

  public void run() {
    try {
      real_run();
    } catch (Exception e) {
      System.err.println("Exception in TercomRunner.run(): " + e.getMessage());
      System.exit(99905);
    }
  }

}

