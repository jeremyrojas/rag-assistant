package com.jeremydevz.rag.assistant.service;

import com.theokanning.openai.embedding.EmbeddingRequest;
import com.theokanning.openai.embedding.EmbeddingResult;
import com.theokanning.openai.service.OpenAiService;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class DocumentService {

    private static final Logger LOGGER = LoggerFactory.getLogger(DocumentService.class);

    private final OpenAiService openAiService;
    private final PineconeService pineconeService;

    @Value("${upload.dir}")
    private String uploadDir;

    // Constants for chunking strategy
    private static final int CHUNK_SIZE = 1000;
    private static final int OVERLAP_SIZE = 200;
    private static final int MIN_CHUNK_SIZE = 20;
    private static final int BATCH_SIZE = 10; // Number of embeddings to process in one batch

    public DocumentService(OpenAiService openAiService, PineconeService pineconeService) {
        this.openAiService = openAiService;
        this.pineconeService = pineconeService;
    }

    /**
     * Process a document by extracting text, chunking it, creating embeddings, and storing in Pinecone.
     * Uses batch processing for efficiency.
     *
     * @param file The document file to process
     * @throws IOException If there's an error reading the file
     */
    public void processDocument(MultipartFile file) throws IOException {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("File cannot be empty");
        }

        LOGGER.info("Processing document: {}", file.getOriginalFilename());

        // Extract text from document
        String text = extractText(file);
        if (text.trim().isEmpty()) {
            LOGGER.warn("Document appears to be empty: {}", file.getOriginalFilename());
            throw new IllegalArgumentException("Document contains no extractable text");
        }

        // Split into chunks with improved chunking strategy
        List<String> chunks = chunkText(text);
        LOGGER.info("Document split into {} chunks", chunks.size());

        if (chunks.isEmpty()) {
            LOGGER.warn("No valid chunks generated from document: {}", file.getOriginalFilename());
            return;
        }

        // Generate a document ID
        String documentId = UUID.randomUUID().toString();

        // Process chunks in batches for efficiency
        for (int i = 0; i < chunks.size(); i += BATCH_SIZE) {
            // Determine batch end (exclusive)
            int endIndex = Math.min(i + BATCH_SIZE, chunks.size());
            List<String> batchChunks = chunks.subList(i, endIndex);

            processChunkBatch(documentId, file.getOriginalFilename(), batchChunks, i);
        }

        LOGGER.info("Successfully processed document: {}", file.getOriginalFilename());
    }

    /**
     * Process a batch of chunks to create embeddings and store them in Pinecone.
     *
     * @param documentId Base document ID
     * @param fileName Original file name
     * @param chunks List of text chunks to process
     * @param startIndex Starting index of the first chunk in this batch
     */
    private void processChunkBatch(String documentId, String fileName, List<String> chunks, int startIndex) {
        try {
            // Prepare lists to hold batch data
            List<String> documentIds = new ArrayList<>();
            List<String> chunkTexts = new ArrayList<>();
            List<List<Double>> embeddings = new ArrayList<>();
            List<Map<String, String>> metadataList = new ArrayList<>();

            // Process each chunk in the batch
            for (int i = 0; i < chunks.size(); i++) {
                String chunk = chunks.get(i);
                int chunkIndex = startIndex + i;

                // Create embedding for this chunk
                EmbeddingRequest embeddingRequest = EmbeddingRequest.builder()
                        .model("text-embedding-ada-002")
                        .input(List.of(chunk))
                        .build();
                EmbeddingResult embeddingResult = openAiService.createEmbeddings(embeddingRequest);
                List<Double> embedding = embeddingResult.getData().get(0).getEmbedding();

                // Create metadata for this chunk
                Map<String, String> metadata = new HashMap<>();
                metadata.put("document_id", documentId);
                metadata.put("document_name", fileName);
                metadata.put("chunk_index", String.valueOf(chunkIndex));
                metadata.put("text", chunk);

                // Add to batch lists
                documentIds.add(documentId);
                chunkTexts.add(chunk);
                embeddings.add(embedding);
                metadataList.add(metadata);
            }

            // Store batch in Pinecone
            pineconeService.storeEmbeddings(documentIds, chunkTexts, embeddings, metadataList);

            LOGGER.debug("Stored batch of {} embeddings for document {}", chunks.size(), documentId);
        } catch (Exception e) {
            LOGGER.error("Error processing chunk batch: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to process document chunks", e);
        }
    }

    /**
     * Extract text content from various file types.
     *
     * @param file The file to extract text from
     * @return The extracted text content
     * @throws IOException If there's an error reading the file
     */
    private String extractText(MultipartFile file) throws IOException {
        // Save the file temporarily
        Path uploadPath = Paths.get(uploadDir);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        Path filePath = uploadPath.resolve(file.getOriginalFilename());
        file.transferTo(filePath.toFile());

        try {
            // Extract text based on file type
            String fileName = file.getOriginalFilename().toLowerCase();

            if (fileName.endsWith(".pdf")) {
                try (PDDocument document = PDDocument.load(filePath.toFile())) {
                    PDFTextStripper stripper = new PDFTextStripper();
                    return stripper.getText(document);
                }
            } else if (fileName.endsWith(".txt") || fileName.endsWith(".md") || fileName.endsWith(".csv")) {
                return Files.readString(filePath);
            } else if (fileName.endsWith(".docx")) {
                // You would need to add the POI library dependency for DOCX support
                // return extractTextFromDocx(filePath.toFile());
                throw new UnsupportedOperationException("DOCX support requires additional dependencies");
            } else {
                throw new UnsupportedOperationException("Unsupported file type: " + fileName);
            }
        } finally {
            // Clean up temporary file
            try {
                Files.deleteIfExists(filePath);
            } catch (IOException e) {
                LOGGER.warn("Failed to delete temporary file: {}", filePath, e);
            }
        }
    }

    /**
     * Chunk text with overlap to preserve context between chunks.
     *
     * @param text The text to chunk
     * @return List of text chunks
     */
    private List<String> chunkText(String text) {
        List<String> chunks = new ArrayList<>();

        // If text is smaller than chunk size, just return it as a single chunk
        if (text.length() <= CHUNK_SIZE) {
            chunks.add(text);
            return chunks;
        }

        // Split by paragraphs first for more meaningful chunks
        String[] paragraphs = text.split("\\n\\s*\\n");
        StringBuilder currentChunk = new StringBuilder();

        for (String paragraph : paragraphs) {
            // If adding this paragraph would exceed chunk size
            if (currentChunk.length() + paragraph.length() > CHUNK_SIZE && currentChunk.length() > 0) {
                // Save current chunk and start a new one with overlap
                chunks.add(currentChunk.toString());

                // Create overlap by taking the end of the previous chunk
                if (currentChunk.length() > OVERLAP_SIZE) {
                    String overlapText = currentChunk.substring(currentChunk.length() - OVERLAP_SIZE);
                    currentChunk = new StringBuilder(overlapText);
                } else {
                    currentChunk = new StringBuilder();
                }
            }

            // Add paragraph to current chunk
            if (currentChunk.length() > 0) {
                currentChunk.append("\n\n");
            }
            currentChunk.append(paragraph);
        }

        // Add the last chunk if it's not empty
        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString());
        }

        // Filter out chunks that are too small
        return chunks.stream()
                .filter(chunk -> chunk.trim().length() > MIN_CHUNK_SIZE)
                .collect(Collectors.toList());
    }
}