#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

#define SERVER_PORT 5432
#define MAX_PENDING 5
#define MAX_LINE 256
#define telemetry 1

int checkFileExists(char *fileName){
    FILE *fp;

    fp = fopen(fileName, "rb");
    if(telemetry){ printf("checking if file exists: %s\n", fileName); }
    if(fp == NULL){
      
        fclose(fp);
        return 0;
    }else{
      
        fclose(fp);
        return 1;
    }

}


int sendFileToSocket(int s, char *fileName){
    FILE *fp;
    char buf[MAX_LINE];
    int n;
    fp = fopen(fileName, "rb");
    if(fp == NULL){
        printf("File does not exist now!!\n");
        return 0;
    }else{
        printf("Sending file.\n");
        while(fgets(buf, sizeof(buf), fp) != NULL){
            n = send(s, buf, strlen(buf), 0);
            printf("Sending: %s", buf);
            if(n < 0){
                perror("send error");
                exit(1);
            }
        }
        // close file
        fclose(fp);
        return 1;
    }
}


int main(int argc, char * argv[]){

  char *host;
  int port;

  struct sockaddr_in sin;
  char buf[MAX_LINE];
  socklen_t len;
  int s, new_s;
  char str[INET_ADDRSTRLEN];

  /* Getting value from argument */
  // if (argc==2) {
  //   host = argv[1];
  // }
  // else {
  //   fprintf(stderr, "usage: %s host\n", argv[0]);
  //   exit(1);
  // }

  if(argc==3){
    host = argv[1];
    port = atoi(argv[2]);
  }else if (argc==2) {
    // fprintf(stderr, "usage: %s ${host} ${port}\n", argv[0]);
    // exit(1);
    host = "0.0.0.0";// argv[1];
    port = atoi(argv[1]);
  }
  else {
    fprintf(stderr, "usage: %s host\n", argv[0]);
    exit(1);
  }


  /* build address data structure */
  bzero((char *)&sin, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_addr.s_addr = inet_addr(host); //INADDR_ANY;
  sin.sin_port = htons(port); // SERVER_PORT
  
  /* setup passive open */
  if ((s = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    perror("simplex-talk: socket");
    exit(1);
  }
 
  inet_ntop(AF_INET, &(sin.sin_addr), str, INET_ADDRSTRLEN);
  printf("Server is using address %s and port %d.\n", str, port);  // SERVER_PORT

  if ((bind(s, (struct sockaddr *)&sin, sizeof(sin))) < 0) {
    perror("simplex-talk: bind");
    exit(1);
  }
  else
    printf("Server bind done.\n");

  listen(s, MAX_PENDING);
  
  /* wait for connection, then receive and print text */
  while(1) {
    if ((new_s = accept(s, (struct sockaddr *)&sin, &len)) < 0) {
      perror("simplex-talk: accept");
      exit(1);
    }
    printf("Server Listening.\n");
    send(new_s, "Hello\n", 6, 0);

    while (len = recv(new_s, buf, sizeof(buf), 0)){
      printf("Server received %d bytes.\n", len);
      if(strncmp(buf, "Bye\0", 4) == 0){
        printf("Client Disconnected.\n");
        break;
      }
      fputs(buf, stdout);
      printf("\nmessage content end!\n");
      char fileName[MAX_LINE];
      
      // sscanf(buf, "%s", fileName);
      // copy buf to fileName without last character
      strncpy(fileName, buf, strlen(buf)-1);
      fileName[strlen(buf)-1] = '\0';

      // print filename as character array
      printf("File name: '%s'\n", fileName);

      // check if file exists
      if (checkFileExists(fileName)){
        if(telemetry){ printf("File exists.\n"); }
        send(new_s, "OK", 3, 0);
        // send(new_s, "1\0", 2, 0);
        int success = sendFileToSocket(new_s, fileName);
        if(success){
          printf("File Sent.");
        }else{
          printf("Unable to send.");
        }
      }else{
        if(telemetry){ printf("File not found\n"); }
        send(new_s, "File not found", 15, 0);
        break;
      }
    }
    close(new_s);
  }
}

