package com.jeremydevz.rag.assistant.service;

import com.google.protobuf.Struct;
import com.jeremydevz.rag.assistant.dto.QueryResponse;
import com.theokanning.openai.service.OpenAiService;
import io.pinecone.proto.ScoredVector;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Slf4j
public class RagService {

    private static final Logger LOGGER = LoggerFactory.getLogger(PineconeService.class);

    private final OpenAiRequestService openAiRequestService;
    private final PineconeService pineconeService;

    public RagService(OpenAiService openAiService, OpenAiRequestService openAiRequestService, PineconeService pineconeService) {
        this.openAiRequestService = openAiRequestService;
        this.pineconeService = pineconeService;
    }

    // Legacy method - kept for backward compatibility
    public String generateAnswer(String question) {
        try {
            // Generate embedding for the question
            List<Double> questionEmbedding = openAiRequestService.createEmbedding(question);

            // Retrieve relevant document chunks
            List<ScoredVector> matches = pineconeService.queryEmbeddings(questionEmbedding, 3);

            // Build context from retrieved chunks
            String context = matches.stream()
                    .sorted(Comparator.comparing(ScoredVector::getScore).reversed())
                    .map(match -> match.getMetadata().getFieldsMap().get("text").getStringValue())
                    .collect(Collectors.joining("\n\n"));

            // Generate response with context-enhanced prompt
            return openAiRequestService.generateChatResponse(context, question);
        } catch (Exception e) {
            LOGGER.error("Error in generateAnswer: {}", e.getMessage());
            throw new RuntimeException("Failed to generate answer", e);
        }
    }

    // Enhanced method with detailed response including sources
    public QueryResponse generateDetailedAnswer(String question) {
        try {
            long startTime = System.currentTimeMillis();
            LOGGER.info("Processing question: {}", question);

            // Generate embedding for the question
            List<Double> questionEmbedding = openAiRequestService.createEmbedding(question);

            // Retrieve relevant document chunks
            List<ScoredVector> matches = pineconeService.queryEmbeddings(questionEmbedding, 3);
            LOGGER.debug("Retrieved {} relevant chunks", matches.size());

            // Extract source documents and build context
            List<String> sourcesUsed = new ArrayList<>();
            StringBuilder contextBuilder = new StringBuilder();

            for (ScoredVector match : matches) {
                Struct metadata = match.getMetadata();
                String chunk = metadata.getFieldsMap().get("text").getStringValue();
                String documentName = metadata.getFieldsMap().get("document_name").getStringValue();
                contextBuilder.append(chunk).append("\n\n");

                if (!sourcesUsed.contains(documentName)) {
                    sourcesUsed.add(documentName);
                }
            }

            String context = contextBuilder.toString();

            // Generate response with context-enhanced prompt
            String answer = openAiRequestService.generateChatResponse(context, question);

            long processingTime = System.currentTimeMillis() - startTime;
            LOGGER.info("Generated answer in {}ms", processingTime);

            QueryResponse response = new QueryResponse();
            response.setAnswer(answer);
            response.setSourcesUsed(sourcesUsed);
            response.setProcessingTimeMs(processingTime);
            return response;

        } catch (Exception e) {
            LOGGER.error("Error in generateDetailedAnswer: {}", e.getMessage());
            throw new RuntimeException("Failed to generate detailed answer", e);
        }
    }
}
