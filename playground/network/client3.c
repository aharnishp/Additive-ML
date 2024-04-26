#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>


#define SERVER_PORT 5432
#define MAX_LINE 256
#define EVER ;;


int main(int argc, char * argv[]){
  FILE *fp;
  struct hostent *hp;
  struct sockaddr_in sin;
  char *host;
  int port;

  char buf[MAX_LINE];
  int s;
  int len;
  if(argc==3){
    host = argv[1];
    port = atoi(argv[2]);
  }else if (argc==2) {
    fprintf(stderr, "usage: %s ${host} ${port}\n", argv[0]);
    exit(1);
    // host = argv[1];
  }
  else {
    fprintf(stderr, "usage: %s host\n", argv[0]);
    exit(1);
  }
  
  /* translate host name into peer's IP address */
  hp = gethostbyname(host);
  if (!hp) {
    fprintf(stderr, "%s: unknown host: %s\n", argv[0], host);
    exit(1);
  }
  else
    printf("Client's remote host: %s\n", argv[1]);
  
  /* build address data structure */
  bzero((char *)&sin, sizeof(sin));
  sin.sin_family = AF_INET;
  bcopy(hp->h_addr, (char *)&sin.sin_addr, hp->h_length);
  sin.sin_port = htons(port);
  
  /* active open */
  if ((s = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    perror("simplex-talk: socket");
    exit(1);
  }
  else
    printf("Client created socket.\n");

  if (connect(s, (struct sockaddr *)&sin, sizeof(sin)) < 0)
    {
      perror("simplex-talk: connect");
      close(s);
      exit(1);
    }
  else
    printf("Client connected.\n");
  
    int len1 = recv(s, buf, 256, 0);
    printf("Server Sent: %s\n", buf);
    printf("Enter file name: ");

  // send and receive  
  while (fgets(buf, sizeof(buf), stdin)) {
    buf[MAX_LINE-1] = '\0';
    // check if "exit" was written
    if(strncmp(buf, "Bye\n",4) == 0){
      send(s, "Bye\0", 4, 0);
      printf("Exiting...\n");

      close(s);
      break;
    }

    len = strlen(buf) + 1;
    send(s, buf, len, 0);
    printf("File requested: %s", buf);

    printf("Waiting for file existence response...\n");
    bzero(buf, sizeof(buf));
    recv(s, buf, 1, 0);
    printf("'%s'\n", buf);

    // if(buf[0] == ''){
    if(buf[0] == 'O'){
    // if(strncmp(buf,"OK",2)){
      printf("File exists.\n");

      // clearing recv buffer of the socket
      recv(s, buf, MAX_LINE, 0);
      bzero(buf, MAX_LINE);

      // return 1;
    }else{
      // clearing recv buffer of the socket
      recv(s, buf, MAX_LINE, 0);
      bzero(buf, MAX_LINE);
      printf("File not found\n");
      break;
    }

    printf("Waiting for contents.\n");

    // receive file contents using recv and stop when the server stops sending
    int recvLen = 0;
    FILE *fp;
    fp = fopen("received.txt", "w");



    while (recvLen = recv(s, buf, sizeof(buf), 0)){

      fwrite(buf, 1, recvLen, fp);
      // printf("Received %d bytes: %s\n", recvLen, buf);
      bzero(buf, sizeof(buf));
      if(recvLen==0 || recvLen!=MAX_LINE){
        break;
      } 
    }
    // close the file
    fclose(fp);
    printf("File received.\n");

    // open file and print contents
    // FILE *fpw;
    // fpw = fopen("received.txt", "r");


    FILE *fpw;
    char buf[MAX_LINE];
    int n;
    fpw = fopen("received.txt", "r");
    if(fpw == NULL){
        printf("File does not exist now!!\n");
    }else{
        printf("---Start of File---\n");
        while(fgets(buf, sizeof(buf), fpw) != NULL){
            // n = send(s, buf, strlen(buf), 0);
            fputs(buf, stdout);
            if(n < 0){
                perror("File read error");
                exit(1);
            }
        }
        // close file
        fclose(fpw);
    }
    



    printf("\n---End of File---\n");
    printf("\nEnter file name: ");

  }
  close(s);
  return 0;
}
