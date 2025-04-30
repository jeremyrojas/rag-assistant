package com.jeremydevz.rag.assistant.config;

import com.theokanning.openai.service.OpenAiService;
import io.pinecone.clients.Index;
import io.pinecone.clients.Pinecone;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenAiConfig {

    @Value("${openai.api.key}")
    private String openaiApiKey;

    @Value("${pinecone.api.key}")
    private String pineconeApiKey;

    @Value("${pinecone.index.name}")
    private String pineconeIndexName;

    @Bean
    public OpenAiService openAiService() {
        return new OpenAiService(openaiApiKey);
    }

    @Bean
    public Index pineconeIndex() {
        Pinecone pineconeClient = new Pinecone.Builder(pineconeApiKey).build();
        return pineconeClient.getIndexConnection(pineconeIndexName);
    }

}
