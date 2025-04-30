package com.jeremydevz.rag.assistant.service;

import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.pinecone.clients.Index;
import io.pinecone.clients.Pinecone;
import io.pinecone.proto.ScoredVector;
import io.pinecone.proto.UpsertResponse;
import io.pinecone.unsigned_indices_model.QueryResponseWithUnsignedIndices;
import io.pinecone.unsigned_indices_model.ScoredVectorWithUnsignedIndices;
import io.pinecone.unsigned_indices_model.VectorWithUnsignedIndices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

import static io.pinecone.commons.IndexInterface.buildUpsertVectorWithUnsignedIndices;

@Service
public class PineconeService {

    private static final Logger LOGGER = LoggerFactory.getLogger(PineconeService.class);

    private final Index pineconeIndex;

    public PineconeService(
            @org.springframework.beans.factory.annotation.Value("${pinecone.api.key}") String apiKey,
            @org.springframework.beans.factory.annotation.Value("${pinecone.index.name}") String indexName) {

        try {
            // Initialize Pinecone client using the Builder pattern
            Pinecone pineconeClient = new Pinecone.Builder(apiKey).build();

            // Get connection to the index
            this.pineconeIndex = pineconeClient.getIndexConnection(indexName);

            LOGGER.info("Successfully connected to Pinecone index: {}", indexName);
        } catch (Exception e) {
            LOGGER.error("Failed to connect to Pinecone: {}", e.getMessage());
            throw new RuntimeException("Could not initialize Pinecone client", e);
        }
    }

    /**
     * Stores multiple embeddings in the Pinecone vector database.
     * Each embedding represents a chunk of text from a document.
     *
     * @param documentIds List of document identifiers (one per chunk, usually the same ID for all chunks from one document)
     * @param chunks List of text chunks corresponding to each embedding
     * @param embeddings List of vector embeddings (each is a list of Double values)
     * @param metadataList List of metadata maps for each embedding (contains info like document name, chunk index, text content)
     */
    public void storeEmbeddings(List<String> documentIds, List<String> chunks,
                                List<List<Double>> embeddings, List<Map<String, String>> metadataList) {
        try {
            List<VectorWithUnsignedIndices> vectors = new ArrayList<>(documentIds.size());

            for (int i = 0; i < documentIds.size(); i++) {
                String vectorId = documentIds.get(i) + "-" + UUID.randomUUID().toString();

                // Convert Double list to Float list (Pinecone requires float values)
                List<Float> floatEmbedding = embeddings.get(i).stream()
                        .map(Double::floatValue)
                        .collect(Collectors.toList());

                // Convert metadata Map to Protobuf Struct
                Struct.Builder structBuilder = Struct.newBuilder();
                metadataList.get(i).forEach((key, value) ->
                        structBuilder.putFields(key, com.google.protobuf.Value.newBuilder().setStringValue(value).build()));

                // Build vector with metadata
                VectorWithUnsignedIndices vector = buildUpsertVectorWithUnsignedIndices(
                        vectorId,
                        floatEmbedding,
                        null, // sparse indices - not used for dense vectors
                        null, // sparse values - not used for dense vectors
                        structBuilder.build()
                );

                vectors.add(vector);
            }

            // Batch upsert to Pinecone
            UpsertResponse response = pineconeIndex.upsert(vectors, "");

            LOGGER.debug("Stored {} embeddings. Upserted count: {}",
                    vectors.size(), response.getUpsertedCount());
        } catch (Exception e) {
            LOGGER.error("Error storing embeddings batch: {}", e.getMessage());
            throw new RuntimeException("Failed to store embeddings batch in Pinecone", e);
        }
    }

    /**
     * Queries the Pinecone database to find the most similar vectors to the query embedding.
     *
     * @param queryEmbedding The embedding vector of the query text
     * @param topK The number of most similar vectors to return
     * @return List of scored vectors
     */
    public List<ScoredVector> queryEmbeddings(List<Double> queryEmbedding, int topK) {
        try {
            // Convert Double list to Float list
            List<Float> floatEmbedding = queryEmbedding.stream()
                    .map(Double::floatValue)
                    .collect(Collectors.toList());

            // Query Pinecone with parameters in the correct order
            QueryResponseWithUnsignedIndices response = pineconeIndex.query(
                    topK,                // topK - first parameter
                    floatEmbedding,      // vector - second parameter
                    null,                // sparseIndices - third parameter
                    null,                // sparseValues - fourth parameter
                    null,                // id - fifth parameter
                    "",                  // namespace - sixth parameter
                    null,                // filter - seventh parameter
                    true,                // includeValues - eighth parameter
                    true                 // includeMetadata - ninth parameter
            );


            // List to hold the final ScoredVector objects
            List<ScoredVector> scoredVectors = new ArrayList<>();

            // Use getMatchesList() instead - this returns the complete list of matches
            for (ScoredVectorWithUnsignedIndices match : response.getMatchesList()) {
                // Extract metadata fields from Struct
                Struct metadata = match.getMetadata();
                Map<String, String> metadataMap = new HashMap<>();
                if (metadata != null) {
                    for (Map.Entry<String, Value> entry : metadata.getFieldsMap().entrySet()) {
                        metadataMap.put(entry.getKey(), entry.getValue().getStringValue());
                    }
                }

                // Convert metadataMap to Struct
                Struct.Builder structBuilder = Struct.newBuilder();
                for (Map.Entry<String, String> entry : metadataMap.entrySet()) {
                    structBuilder.putFields(entry.getKey(), Value.newBuilder().setStringValue(entry.getValue()).build());
                }
                Struct finalMetadata = structBuilder.build();

                // Create a ScoredVector using the builder
                ScoredVector scoredVector = ScoredVector.newBuilder()
                        .setScore(match.getScore())    // Set the score
                        .setMetadata(finalMetadata)    // Set the Struct metadata
                        .build();

                // Add the ScoredVector to the result list
                scoredVectors.add(scoredVector);
            }

            LOGGER.debug("Retrieved {} matches from Pinecone", scoredVectors.size());
            return scoredVectors;
        } catch (Exception e) {
            LOGGER.error("Error querying embeddings: {}", e.getMessage());
            throw new RuntimeException("Failed to query embeddings from Pinecone", e);
        }
    }

}