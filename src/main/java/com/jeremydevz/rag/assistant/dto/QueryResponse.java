package com.jeremydevz.rag.assistant.dto;

import java.util.List;

/**
 * Data Transfer Object (DTO) for responses from the RAG service.
 * Contains the answer, sources used, and processing time.
 */
public class QueryResponse {
    private String answer;
    private List<String> sourcesUsed;
    private long processingTimeMs;

    // Default constructor
    public QueryResponse() {
    }

    // All-args constructor
    public QueryResponse(String answer, List<String> sourcesUsed, long processingTimeMs) {
        this.answer = answer;
        this.sourcesUsed = sourcesUsed;
        this.processingTimeMs = processingTimeMs;
    }

    // Getters and setters
    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }

    public List<String> getSourcesUsed() {
        return sourcesUsed;
    }

    public void setSourcesUsed(List<String> sourcesUsed) {
        this.sourcesUsed = sourcesUsed;
    }

    public long getProcessingTimeMs() {
        return processingTimeMs;
    }

    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }
}