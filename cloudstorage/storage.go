// Provides functionalities to use GCP Cloud Storage as remote model store
package cloudstorage

import (
	"context"
	"io"
	"os"

	"cloud.google.com/go/storage"
	"github.com/DHBWMannheim/ml-server/util"
	"google.golang.org/api/option"
)

type Storage interface {
	// Downloads the model based on the full path provided
	// in the `name` parameter.
	//
	// It returns the path, where the models lives. In case
	// the model was not present, it returns an storage.ErrObjNotFound.
	DownloadModel(ctx context.Context, name, modelPath string) (string, error)
}

type cloudStorage struct {
	bucket *storage.BucketHandle
}

func NewCloudStorageService(ctx context.Context, bucketName string) Storage {
	client, err := storage.NewClient(ctx, option.WithoutAuthentication(), option.WithScopes(storage.ScopeReadOnly))
	if err != nil {
		panic(err)
	}

	c := &cloudStorage{}

	c.bucket = client.Bucket(bucketName)

	return c
}

// Downloads the model based on the full path provided
// in the `name` parameter.
//
// It returns the path, where the models lives. In case
// the model was not present, it returns an storage.ErrObjNotFound.
func (c *cloudStorage) DownloadModel(ctx context.Context, name, modelPath string) (string, error) {
	rc, err := c.bucket.Object(name).NewReader(ctx)
	if err != nil {
		return "", err
	}

	defer rc.Close()

	f, err := os.CreateTemp("", "*.zip")
	defer os.Remove(f.Name())
	if err != nil {
		return "", err
	}

	defer f.Close()

	io.Copy(f, rc)
	if err := util.ExtractTfArchive(f, modelPath); err != nil {
		return "", err
	}
	return modelPath, nil
}
