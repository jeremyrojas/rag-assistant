package com.jeremydevz.rag.assistant.service;

import com.theokanning.openai.completion.chat.ChatCompletionRequest;
import com.theokanning.openai.completion.chat.ChatCompletionResult;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.completion.chat.ChatMessageRole;
import com.theokanning.openai.embedding.EmbeddingRequest;
import com.theokanning.openai.embedding.EmbeddingResult;
import com.theokanning.openai.service.OpenAiService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Service
public class OpenAiRequestService {

    private static final Logger LOGGER = LoggerFactory.getLogger(OpenAiRequestService.class);

    private final OpenAiService openAiService;

    public OpenAiRequestService(@Value("${openai.api.key}") String apiKey) {
        this.openAiService = new OpenAiService(apiKey);
    }

    public List<Double> createEmbedding(String text) {
        try {
            EmbeddingRequest request = EmbeddingRequest.builder()
                    .model("text-embedding-ada-002")
                    .input(Collections.singletonList(text))
                    .build();

            EmbeddingResult result = openAiService.createEmbeddings(request);
            return result.getData().get(0).getEmbedding();
        } catch (Exception e) {
            LOGGER.error("Error creating embedding: {}", e.getMessage());
            throw new RuntimeException("Failed to create embedding", e);
        }
    }

    public String generateChatResponse(String context, String question) {
        try {
            ChatCompletionRequest request = ChatCompletionRequest.builder()
                    .model("gpt-3.5-turbo")
                    .messages(Arrays.asList(
                            new ChatMessage(ChatMessageRole.SYSTEM.value(), "You are a helpful assistant that answers questions based on the provided context."),
                            new ChatMessage(ChatMessageRole.USER.value(), "Context: " + context + "\n\nQuestion: " + question + "\n\nAnswer:")
                    ))
                    .maxTokens(500)
                    .build();

            ChatCompletionResult response = openAiService.createChatCompletion(request);
            return response.getChoices().get(0).getMessage().getContent();
        } catch (Exception e) {
            LOGGER.error("Error generating chat response: {}", e.getMessage());
            throw new RuntimeException("Failed to generate chat response", e);
        }
    }
}