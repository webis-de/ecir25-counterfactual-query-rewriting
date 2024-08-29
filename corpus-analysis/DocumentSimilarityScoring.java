import de.webis.copycat.DocumentResolver;
import de.webis.copycat_cli.CliArguments;
import de.webis.copycat.document_preprocessing.PreprocessingArgs;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.codehaus.jackson.map.ObjectMapper;
import net.sourceforge.argparse4j.inf.Namespace;

import de.webis.cikm20_duplicates.spark.SparkEnrichRelevanceTransferPairs;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import de.webis.trec_ndd.trec_collections.CollectionDocument;

public class DocumentSimilarityScoring implements CliArguments {
    public static void main(String[] args) throws Exception {
        Namespace parsedArgs = parseArgs(args);
        if(parsedArgs == null) {
            return;
        }
        System.out.println("Hello world");
        List<String> ret = new ArrayList<>();
        allLines(parsedArgs.get("input")).forEach(i -> {
            Map<String, String> docs = (Map) i.get("docs");
            i.remove("docs");
            List<Map<String, Object>> scores = new ArrayList<>();
            
            for(String docA: docs.keySet()) {
                CollectionDocument a = CollectionDocument.collectionDocument(docs.get(docA), docA);
                for(String docB: docs.keySet()) {
                    if(docA.equals(docB)) {
                        continue;
                    }
                    CollectionDocument b = CollectionDocument.collectionDocument(docs.get(docB), docB);
                    
                    Map<String, Object> m = new HashMap<>();
                    m.put("doc-a", docA);
                    m.put("doc-b", docB);
                    m.put("s3-score", SparkEnrichRelevanceTransferPairs.s3Score(a, b));
                    scores.add(m);
                }
            }
            i.put("scores", scores);
            try {
                ret.add(new ObjectMapper().writeValueAsString(i));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        
        String t = "";
        for (String i: ret) {
            t += "\n" + i;
        }

        Files.write(Paths.get((String) parsedArgs.get("output")), t.trim().getBytes());
    }

    private static Stream<Map<String, Object>> allLines(String path) {
        try {
            return Files.lines(Paths.get(path)).map(i -> parseJson(i));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static Map<String, Object> parseJson(String json) {
        try {
            return new ObjectMapper().readValue(json, Map.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static Namespace parseArgs(String[] args) {
        ArgumentParser parser = argParser();

        try {
            return parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            return null;
        }
    }

    static ArgumentParser argParser() {
        ArgumentParser ret = ArgumentParsers.newFor("DocumentSimilarityScoring").build();
        ret.addArgument("--input")
            .help("The jsonl file with the entries, each line like {\"url\": \"http://url.com\", \"docs\": {\"doc1\": \"some text\", \"doc2\": \"some text\"}}.")
            .required(true);
        ret.addArgument("--output")
            .help("The jsonl file to write.")
            .type(String.class)
            .required(true);

        PreprocessingArgs.addArgs(ret);
        return ret;
    }
}

