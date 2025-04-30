package com.jeremydevz.rag.assistant.controller;

import com.jeremydevz.rag.assistant.dto.QueryResponse;
import com.jeremydevz.rag.assistant.service.DocumentService;
import com.jeremydevz.rag.assistant.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // For development - restrict in production
public class RagController {

    private static final Logger LOGGER = LoggerFactory.getLogger(RagController.class);

    private final DocumentService documentService;
    private final RagService ragService;

    public RagController(DocumentService documentService, RagService ragService) {
        this.documentService = documentService;
        this.ragService = ragService;
    }

    @PostMapping("/upload")
    public ResponseEntity<String> uploadDocument(@RequestParam("file") MultipartFile file) {
        try {
            // Validate input
            if (file == null || file.isEmpty()) {
                return ResponseEntity.badRequest().body("File cannot be empty");
            }

            documentService.processDocument(file);
            return ResponseEntity.ok("Document processed successfully");
        } catch (Exception e) {
            LOGGER.error("Error processing document", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error processing document: " + e.getMessage());
        }
    }

    // Enhanced endpoint with detailed response
    @PostMapping("/ask")
    public ResponseEntity<?> askQuestion(@RequestParam("question") String question) {
        try {
            if (question == null || question.trim().isEmpty()) {
                return ResponseEntity.badRequest().body("Question cannot be empty");
            }

            QueryResponse response = ragService.generateDetailedAnswer(question);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            LOGGER.error("Error generating answer", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error generating answer: " + e.getMessage());
        }
    }

    // Legacy endpoint for backward compatibility
    @PostMapping("/simple-ask")
    public ResponseEntity<String> askSimpleQuestion(@RequestParam("question") String question) {
        try {
            if (question == null || question.trim().isEmpty()) {
                return ResponseEntity.badRequest().body("Question cannot be empty");
            }

            String answer = ragService.generateAnswer(question);
            return ResponseEntity.ok(answer);
        } catch (Exception e) {
            LOGGER.error("Error generating simple answer", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error generating answer: " + e.getMessage());
        }
    }
}
